"""Verification suite for the slice's content.

Confirms the reference solution, capsule example, and warm-up solution actually
run in the pinned environment, and that every published item carries sources and
a verified status. This is the executable half of "documentation-grounded".
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "apps" / "api"))

from app import content, executor_service  # noqa: E402

pytestmark = pytest.mark.usefixtures("docker_or_skip")

CH = content.load_challenge("tensor-row-normalize-001")
CAP = content.load_capsule("reduction-dimensions-keepdim")
WARM = content.load_warmup("row-sums-keepdim-warmup")


def test_published_items_are_verified_with_sources():
    assert CH.verification.status == "verified"
    assert CAP.verification.status == "verified"
    assert CH.sources and CAP.sources
    registry = content.load_sources()
    for sid in set(CH.sources) | set(CAP.sources):
        assert sid in registry, f"source '{sid}' missing from registry"
        assert registry[sid].url.startswith("https://")


def test_reference_solution_passes_visible_and_hidden():
    solution = content.load_solution_code(CH)
    checks = []
    for t in CH.visible_tests:
        d = t.model_dump(); d["visibility"] = "visible"; checks.append(d)
    checks += content.load_hidden_checks(CH)
    res = executor_service.run_function_checks(CH.function_name, solution, checks)
    assert res.ok, res.error
    failed = [c for c in res.result["checks"] if not c["passed"]]
    assert not failed, f"reference solution failed checks: {failed}"


def test_capsule_example_runs_clean():
    code = content.load_example_code(CAP)
    res = executor_service.run_script(code)
    assert res.ok and res.result["error"] is None
    assert "(2, 1)" in res.result["stdout"]  # keepdim shape demonstrated
    assert res.result["stderr"] == ""        # numpy warning is gone in the pinned image


def test_warmup_inline_solution_passes():
    solution = WARM.starter_code.replace("        pass", "        " + WARM.solution_inline.strip())
    # Build a clean function body from starter + inline solution body.
    body = f"import torch\n\n\ndef {WARM.function_name}(x):\n    {WARM.solution_inline.strip()}\n"
    checks = []
    for t in WARM.visible_tests:
        d = t.model_dump(); d["visibility"] = "visible"; checks.append(d)
    res = executor_service.run_function_checks(WARM.function_name, body, checks)
    assert res.ok, res.error
    assert all(c["passed"] for c in res.result["checks"]), res.result["checks"]
