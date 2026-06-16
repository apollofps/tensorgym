"""Sandbox executor tests against the real pinned Docker image.

Asserts the security properties (no network, timeout reaping) and the structured
result protocol. These run only when Docker + the image are available.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "apps" / "api"))

from app import executor_service  # noqa: E402

pytestmark = pytest.mark.usefixtures("docker_or_skip")

GOOD = "import torch\n\ndef normalize_rows(x):\n    return x / x.sum(dim=1, keepdim=True)\n"
BAD = "import torch\n\ndef normalize_rows(x):\n    return x / x.sum(dim=1)\n"


def test_structured_pass_and_metadata():
    checks = [{"name": "t", "visibility": "visible", "kind": "rows_sum_to_one",
               "input": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]}]
    res = executor_service.run_function_checks("normalize_rows", GOOD, checks,
                                               metadata_input=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    assert res.ok and res.result is not None
    assert res.result["checks"][0]["passed"] is True
    md = res.result["metadata"]
    assert md["shape"] == [2, 3] and md["dtype"] == "torch.float32"


def test_missing_keepdim_reports_runtime_exception():
    checks = [{"name": "t", "visibility": "visible", "kind": "rows_sum_to_one",
               "input": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]}]
    res = executor_service.run_function_checks("normalize_rows", BAD, checks)
    c = res.result["checks"][0]
    assert c["passed"] is False
    assert c["category"] == "runtime_exception"
    assert "size of tensor" in c["error"]["message"].lower()


def test_no_loops_static_check():
    looped = ("import torch\n\ndef f(x):\n    out = x.clone()\n"
              "    for i in range(x.shape[0]):\n        out[i] = x[i] / x[i].sum()\n    return out\n")
    checks = [{"name": "nl", "visibility": "hidden", "kind": "no_loops"}]
    res = executor_service.run_function_checks("f", looped, checks)
    assert res.result["checks"][0]["passed"] is False
    assert res.result["checks"][0]["category"] == "forbidden_api"


def test_network_is_disabled():
    # --network none means any outbound connection attempt must fail inside the box.
    code = (
        "import socket\n"
        "def f(x):\n"
        "    s = socket.socket(); s.settimeout(2)\n"
        "    s.connect(('1.1.1.1', 53))\n"  # would succeed if network were allowed
        "    return x\n"
    )
    checks = [{"name": "net", "visibility": "hidden", "kind": "same_shape", "input": [[1.0, 2.0]]}]
    res = executor_service.run_function_checks("f", code, checks)
    c = res.result["checks"][0]
    assert c["passed"] is False
    assert c["category"] == "runtime_exception"
    assert c["error"]["type"] in {"OSError", "socket.gaierror", "gaierror", "TimeoutError"}


def test_timeout_is_enforced():
    code = "def f(x):\n    while True:\n        pass\n"
    checks = [{"name": "loop", "visibility": "hidden", "kind": "same_shape", "input": [[1.0]]}]
    res = executor_service.run_function_checks("f", code, checks, timeout_s=12)
    # Either the in-container watchdog catches it (structured) or the host wall-clock kills it.
    if res.ok:
        assert res.result["checks"][0]["passed"] is False
    else:
        assert res.timed_out is True


def test_script_mode_captures_stdout():
    res = executor_service.run_script("print('hello'); import torch; print(tuple(torch.zeros(2,3).shape))")
    assert res.ok
    assert "hello" in res.result["stdout"]
    assert "(2, 3)" in res.result["stdout"]
