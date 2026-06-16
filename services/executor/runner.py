"""In-container test runner for TensorGym (executes UNTRUSTED learner code).

This file runs INSIDE the sandbox container, never in the API process. It reads a
job description from a read-only mounted file, runs the learner's code, and emits
exactly one structured-JSON line on stdout, prefixed with a sentinel so the host
can isolate it from any library chatter.

Job schema (JSON):
  {
    "mode": "checks" | "script",
    "submission_code": "<learner code>",
    "function_name": "normalize_rows",          # checks mode
    "checks": [ {name, visibility, kind, input?, params?}, ... ],  # checks mode
    "metadata_input": [[...]],                    # optional; collect tensor metadata
    "soft_timeout_s": 8                           # in-runner watchdog
  }

The runner reports FACTS (shapes, dtypes, exceptions). Pedagogical
classification (the mistake taxonomy) is done by the API, not here.
"""

from __future__ import annotations

import ast
import contextlib
import io
import json
import signal
import sys
import traceback
from typing import Any

RESULT_SENTINEL = "__TENSORGYM_RESULT__"
MAX_TEXT = 8000  # cap any captured text to keep output bounded


def _clip(text: str) -> str:
    if len(text) > MAX_TEXT:
        return text[:MAX_TEXT] + f"... [truncated {len(text) - MAX_TEXT} chars]"
    return text


class _Timeout(Exception):
    pass


def _install_watchdog(seconds: int) -> None:
    def _handler(signum: int, frame: Any) -> None:  # noqa: ANN401
        raise _Timeout(f"execution exceeded {seconds}s")

    signal.signal(signal.SIGALRM, _handler)
    signal.alarm(max(1, int(seconds)))


# --------------------------------------------------------------------------- #
# Loading learner code
# --------------------------------------------------------------------------- #
def _load_namespace(code: str) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Compile+exec the submission in a fresh namespace. Returns (ns, error)."""
    ns: dict[str, Any] = {}
    try:
        compiled = compile(code, "<submission>", "exec")
    except SyntaxError as exc:
        return None, {
            "type": "SyntaxError",
            "message": _clip(f"{exc.msg} (line {exc.lineno})"),
            "phase": "compile",
        }
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            exec(compiled, ns)  # noqa: S102 - sandboxed by the container
    except Exception as exc:  # noqa: BLE001
        return None, {
            "type": type(exc).__name__,
            "message": _clip(str(exc)),
            "phase": "exec",
            "traceback": _clip(traceback.format_exc()),
        }
    return ns, None


def _has_python_loops(code: str) -> bool:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    return any(isinstance(node, (ast.For, ast.While, ast.AsyncFor)) for node in ast.walk(tree))


# --------------------------------------------------------------------------- #
# Tensor helpers
# --------------------------------------------------------------------------- #
def _tensor_metadata(t: Any) -> dict[str, Any]:  # noqa: ANN401
    import torch

    if not isinstance(t, torch.Tensor):
        return {"is_tensor": False, "python_type": type(t).__name__}
    return {
        "is_tensor": True,
        "shape": list(t.shape),
        "rank": t.dim(),
        "dtype": str(t.dtype),
        "device": str(t.device),
        "requires_grad": bool(t.requires_grad),
        "grad_fn": type(t.grad_fn).__name__ if t.grad_fn is not None else None,
        "numel": int(t.numel()),
        "element_size_bytes": t.element_size(),
        "nbytes": int(t.numel() * t.element_size()),
        "is_contiguous": bool(t.is_contiguous()),
    }


def _call(fn: Any, input_spec: Any, *, params: dict | None = None):  # noqa: ANN001, ANN401
    """Build input tensor(s) and call fn.

    input_spec can be:
      - a nested list → single float32 tensor
      - a dict with "args" key → positional args, each converted to a tensor
      - a dict with "kwargs" key → keyword args
    params (from the check spec) can override dtype via "input_dtype".
    """
    import torch

    p = params or {}
    dtype_str = p.get("input_dtype", "float32")
    dtype = getattr(torch, dtype_str, torch.float32)

    if isinstance(input_spec, dict):
        if "args" in input_spec:
            args = [torch.tensor(a, dtype=dtype) if isinstance(a, list) else a for a in input_spec["args"]]
            return args[0] if args else None, fn(*args)
        if "kwargs" in input_spec:
            kw = {k: torch.tensor(v, dtype=dtype) if isinstance(v, list) else v for k, v in input_spec["kwargs"].items()}
            first = next(iter(kw.values()), None)
            return first, fn(**kw)

    x = torch.tensor(input_spec, dtype=dtype)
    return x, fn(x)


# --------------------------------------------------------------------------- #
# Check kinds. Each returns a dict with passed/detail/error/observed/expected.
# --------------------------------------------------------------------------- #
def _check_rows_sum_to_one(fn, spec):  # noqa: ANN001
    import torch

    x, out = _call(fn, spec["input"])
    if not isinstance(out, torch.Tensor):
        return _fail("type", f"function returned {type(out).__name__}, expected a tensor")
    if tuple(out.shape) != tuple(x.shape):
        return _fail(
            "shape",
            f"output shape {tuple(out.shape)} != input shape {tuple(x.shape)}",
            observed={"shape": list(out.shape)},
            expected={"shape": list(x.shape)},
        )
    sums = out.sum(dim=1)
    ones = torch.ones_like(sums)
    if not torch.allclose(sums, ones, atol=1e-5):
        return _fail(
            "value",
            f"each row should sum to 1; got row sums {sums.tolist()}",
            observed={"row_sums": [round(v, 4) for v in sums.tolist()]},
            expected={"row_sums": ones.tolist()},
        )
    return _pass(f"all {x.shape[0]} rows sum to 1")


def _check_same_shape(fn, spec):  # noqa: ANN001
    import torch

    x, out = _call(fn, spec["input"])
    if not isinstance(out, torch.Tensor):
        return _fail("type", f"function returned {type(out).__name__}, expected a tensor")
    if tuple(out.shape) != tuple(x.shape):
        return _fail(
            "shape",
            f"output shape {tuple(out.shape)} != input shape {tuple(x.shape)}",
            observed={"shape": list(out.shape)},
            expected={"shape": list(x.shape)},
        )
    return _pass(f"output shape {tuple(out.shape)} matches input")


def _check_expect_shape(fn, spec):  # noqa: ANN001
    import torch

    want = tuple(spec["params"]["shape"])
    _, out = _call(fn, spec["input"])
    if not isinstance(out, torch.Tensor):
        return _fail("type", f"function returned {type(out).__name__}, expected a tensor")
    if tuple(out.shape) != want:
        return _fail(
            "shape",
            f"output shape {tuple(out.shape)} != expected {want}",
            observed={"shape": list(out.shape)},
            expected={"shape": list(want)},
        )
    return _pass(f"output shape {want} as expected")


def _check_exact_allclose(fn, spec):  # noqa: ANN001
    import torch

    expected = torch.tensor(spec["params"]["expected"], dtype=torch.float32)
    _, out = _call(fn, spec["input"])
    if not isinstance(out, torch.Tensor):
        return _fail("type", f"function returned {type(out).__name__}, expected a tensor")
    if tuple(out.shape) != tuple(expected.shape):
        return _fail(
            "shape",
            f"output shape {tuple(out.shape)} != expected {tuple(expected.shape)}",
            observed={"shape": list(out.shape)},
            expected={"shape": list(expected.shape)},
        )
    if not torch.allclose(out, expected, atol=1e-5):
        return _fail(
            "value",
            "output values do not match expected",
            observed={"values": out.tolist()},
            expected={"values": expected.tolist()},
        )
    return _pass("output matches expected values")


def _check_no_loops(fn, spec, *, code: str):  # noqa: ANN001
    if _has_python_loops(code):
        return _fail("forbidden_api", "Python for/while loop detected; use tensor operations instead")
    return _pass("no Python loops used")


def _check_expect_dtype(fn, spec):  # noqa: ANN001
    import torch

    want = spec["params"]["dtype"]
    _, out = _call(fn, spec["input"], params=spec.get("params"))
    if not isinstance(out, torch.Tensor):
        return _fail("type", f"function returned {type(out).__name__}, expected a tensor")
    got = str(out.dtype)
    if got != f"torch.{want}" and got != want:
        return _fail("dtype", f"output dtype {got}, expected torch.{want}",
                     observed={"dtype": got}, expected={"dtype": f"torch.{want}"})
    return _pass(f"output dtype is {got}")


def _check_expect_requires_grad(fn, spec):  # noqa: ANN001
    import torch

    want = spec["params"]["requires_grad"]
    _, out = _call(fn, spec["input"], params=spec.get("params"))
    if not isinstance(out, torch.Tensor):
        return _fail("type", f"function returned {type(out).__name__}, expected a tensor")
    if out.requires_grad != want:
        return _fail("value", f"requires_grad is {out.requires_grad}, expected {want}",
                     observed={"requires_grad": out.requires_grad}, expected={"requires_grad": want})
    return _pass(f"requires_grad is {out.requires_grad}")


def _check_expect_value(fn, spec):  # noqa: ANN001
    """Check non-tensor return value (int, float, bool, list)."""
    want = spec["params"]["expected"]
    _, out = _call(fn, spec["input"], params=spec.get("params"))
    import torch
    if isinstance(out, torch.Tensor):
        out = out.item() if out.numel() == 1 else out.tolist()
    if isinstance(want, float):
        if abs(out - want) > spec["params"].get("atol", 1e-5):
            return _fail("value", f"returned {out}, expected {want}", observed={"value": out}, expected={"value": want})
    elif out != want:
        return _fail("value", f"returned {out}, expected {want}", observed={"value": out}, expected={"value": want})
    return _pass(f"returned {out}")


def _check_cols_sum_to_one(fn, spec):  # noqa: ANN001
    import torch

    x, out = _call(fn, spec["input"])
    if not isinstance(out, torch.Tensor):
        return _fail("type", f"function returned {type(out).__name__}, expected a tensor")
    if tuple(out.shape) != tuple(x.shape):
        return _fail("shape", f"output shape {tuple(out.shape)} != input shape {tuple(x.shape)}",
                     observed={"shape": list(out.shape)}, expected={"shape": list(x.shape)})
    sums = out.sum(dim=0)
    ones = torch.ones_like(sums)
    if not torch.allclose(sums, ones, atol=1e-5):
        return _fail("value", f"each column should sum to 1; got col sums {sums.tolist()}",
                     observed={"col_sums": [round(v, 4) for v in sums.tolist()]},
                     expected={"col_sums": ones.tolist()})
    return _pass(f"all {x.shape[1]} columns sum to 1")


def _check_output_is_tensor(fn, spec):  # noqa: ANN001
    import torch

    _, out = _call(fn, spec["input"], params=spec.get("params"))
    if not isinstance(out, torch.Tensor):
        return _fail("type", f"function returned {type(out).__name__}, expected a tensor")
    return _pass(f"output is a tensor with shape {tuple(out.shape)}")


def _check_runs_without_error(fn, spec):  # noqa: ANN001
    _call(fn, spec["input"], params=spec.get("params"))
    return _pass("function ran without error")


_CHECKERS = {
    "rows_sum_to_one": _check_rows_sum_to_one,
    "cols_sum_to_one": _check_cols_sum_to_one,
    "same_shape": _check_same_shape,
    "expect_shape": _check_expect_shape,
    "expect_dtype": _check_expect_dtype,
    "expect_requires_grad": _check_expect_requires_grad,
    "expect_value": _check_expect_value,
    "exact_allclose": _check_exact_allclose,
    "no_loops": _check_no_loops,
    "output_is_tensor": _check_output_is_tensor,
    "runs_without_error": _check_runs_without_error,
}


def _pass(detail: str) -> dict[str, Any]:
    return {"passed": True, "detail": detail, "error": None, "observed": None, "expected": None}


def _fail(category: str, detail: str, observed=None, expected=None) -> dict[str, Any]:  # noqa: ANN001
    return {
        "passed": False,
        "category": category,
        "detail": detail,
        "error": None,
        "observed": observed,
        "expected": expected,
    }


def _run_check(fn, spec, code):  # noqa: ANN001
    kind = spec.get("kind")
    checker = _CHECKERS.get(kind)
    base = {"name": spec.get("name"), "visibility": spec.get("visibility", "visible"), "kind": kind}
    if checker is None:
        return {**base, "passed": False, "category": "internal", "detail": f"unknown check kind '{kind}'",
                "error": None, "observed": None, "expected": None}
    try:
        if kind == "no_loops":
            result = checker(fn, spec, code=code)
        else:
            result = checker(fn, spec)
    except _Timeout as exc:
        result = {"passed": False, "category": "timeout", "detail": str(exc),
                  "error": {"type": "Timeout", "message": str(exc)}, "observed": None, "expected": None}
    except Exception as exc:  # noqa: BLE001 - learner code may raise anything
        result = {"passed": False, "category": "runtime_exception", "detail": _clip(str(exc)),
                  "error": {"type": type(exc).__name__, "message": _clip(str(exc).splitlines()[0] if str(exc) else "")},
                  "observed": None, "expected": None}
    return {**base, **result}


# --------------------------------------------------------------------------- #
# Modes
# --------------------------------------------------------------------------- #
def _run_checks_mode(job: dict[str, Any]) -> dict[str, Any]:
    code = job["submission_code"]
    ns, load_err = _load_namespace(code)
    if load_err is not None:
        return {"mode": "checks", "loaded": False, "load_error": load_err, "checks": [], "metadata": None}

    fn = ns.get(job["function_name"]) if ns else None
    if not callable(fn):
        return {
            "mode": "checks", "loaded": False,
            "load_error": {"type": "NameError",
                           "message": f"function '{job['function_name']}' is not defined", "phase": "lookup"},
            "checks": [], "metadata": None,
        }

    checks = [_run_check(fn, spec, code) for spec in job.get("checks", [])]

    metadata = None
    if job.get("metadata_input") is not None:
        try:
            _, out = _call(fn, job["metadata_input"])
            metadata = _tensor_metadata(out)
        except Exception as exc:  # noqa: BLE001
            metadata = {"error": {"type": type(exc).__name__, "message": _clip(str(exc).splitlines()[0] if str(exc) else "")}}

    return {"mode": "checks", "loaded": True, "load_error": None, "checks": checks, "metadata": metadata}


def _run_script_mode(job: dict[str, Any]) -> dict[str, Any]:
    code = job["submission_code"]
    buf_out, buf_err = io.StringIO(), io.StringIO()
    error = None
    try:
        compiled = compile(code, "<snippet>", "exec")
        with contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(buf_err):
            exec(compiled, {})  # noqa: S102 - sandboxed by container
    except _Timeout as exc:
        error = {"type": "Timeout", "message": str(exc)}
    except Exception as exc:  # noqa: BLE001
        error = {"type": type(exc).__name__, "message": _clip(str(exc)), "traceback": _clip(traceback.format_exc())}
    return {"mode": "script", "stdout": _clip(buf_out.getvalue()), "stderr": _clip(buf_err.getvalue()), "error": error}


def main() -> None:
    try:
        job_path = sys.argv[1]
        with open(job_path, encoding="utf-8") as fh:
            job = json.load(fh)
    except Exception as exc:  # noqa: BLE001
        print(RESULT_SENTINEL + json.dumps({"fatal": f"could not read job: {exc}"}))
        return

    _install_watchdog(int(job.get("soft_timeout_s", 8)))
    try:
        result = _run_script_mode(job) if job.get("mode") == "script" else _run_checks_mode(job)
    except _Timeout as exc:
        result = {"fatal": f"timeout: {exc}"}
    except Exception as exc:  # noqa: BLE001
        result = {"fatal": _clip(f"{type(exc).__name__}: {exc}")}
    finally:
        signal.alarm(0)

    print(RESULT_SENTINEL + json.dumps(result))


if __name__ == "__main__":
    main()
