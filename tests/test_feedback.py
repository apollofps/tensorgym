"""Unit tests for the deterministic mistake-taxonomy feedback engine (no Docker)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "apps" / "api"))

from app.content import load_challenge  # noqa: E402
from app.feedback import build_feedback  # noqa: E402

CH = load_challenge("tensor-row-normalize-001")


def _result(checks):
    return {"loaded": True, "load_error": None, "checks": checks, "metadata": None}


def test_all_pass():
    fb = build_feedback(_result([
        {"name": "a", "visibility": "visible", "passed": True},
        {"name": "b", "visibility": "hidden", "passed": True},
    ]), CH)
    assert fb.status == "passed"
    assert fb.passed_count == 2 and fb.total_count == 2


def test_missing_keepdim_runtime_is_broadcasting():
    # The no-keepdim mistake surfaces as a RuntimeError about tensor sizes.
    fb = build_feedback(_result([{
        "name": "two-by-three", "visibility": "visible", "passed": False,
        "category": "runtime_exception",
        "error": {"type": "RuntimeError",
                  "message": "The size of tensor a (3) must match the size of tensor b (2) at non-singleton dimension 1"},
    }]), CH)
    assert fb.status == "failed"
    assert fb.mistake_category == "Broadcasting error"
    assert "keepdim=True" in (fb.next_step or "")
    assert fb.capsule_suggestion == "reduction-dimensions-keepdim"


def test_wrong_dim_value_error():
    fb = build_feedback(_result([{
        "name": "v", "visibility": "visible", "passed": False, "category": "value",
        "observed": {"row_sums": [2.1, 0.9]}, "expected": {"row_sums": [1.0, 1.0]},
        "detail": "row sums wrong",
    }]), CH)
    assert fb.mistake_category == "Numerical error"
    assert "dimension" in (fb.next_step or "").lower()


def test_forbidden_loop():
    fb = build_feedback(_result([{
        "name": "hidden-no-loops", "visibility": "hidden", "passed": False,
        "category": "forbidden_api", "detail": "Python for/while loop detected",
    }]), CH)
    assert fb.mistake_category == "Forbidden API usage"


def test_shape_reduced_dimension():
    fb = build_feedback(_result([{
        "name": "s", "visibility": "visible", "passed": False, "category": "shape",
        "observed": {"shape": [2]}, "expected": {"shape": [2, 3]},
        "detail": "shape mismatch",
    }]), CH)
    assert fb.mistake_category == "Dimension-selection error"


def test_definition_error():
    fb = build_feedback({"loaded": False,
                         "load_error": {"type": "NameError", "message": "function 'normalize_rows' is not defined"}}, CH)
    assert fb.status == "error"
    assert fb.mistake_category == "Runtime exception"


def test_hidden_detail_is_not_leaked():
    fb = build_feedback(_result([{
        "name": "hidden-x", "visibility": "hidden", "passed": False,
        "category": "value", "detail": "row sums wrong: [2.1, 0.9]",
    }]), CH)
    hidden = [r for r in fb.hidden_results if r.name == "hidden-x"][0]
    assert hidden.detail == "hidden check failed"  # internals not exposed
