"""Deterministic feedback engine (mistake taxonomy -> targeted guidance).

Maps the FACTS reported by the sandbox runner to the spec's mistake taxonomy and
produces four-part feedback: what was observed, the concept involved, what to
inspect, and the smallest next step. No solution code is revealed.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from .schemas import Challenge


class CheckResult(BaseModel):
    name: str
    visibility: str
    passed: bool
    detail: str | None = None
    category: str | None = None


class Feedback(BaseModel):
    status: str  # "passed" | "failed" | "error"
    summary: str
    mistake_category: str | None = None
    observations: list[str] = []
    concept: str | None = None
    inspect: str | None = None
    next_step: str | None = None
    capsule_suggestion: str | None = None
    drill_suggestion: str | None = None
    visible_results: list[CheckResult] = []
    hidden_results: list[CheckResult] = []
    passed_count: int = 0
    total_count: int = 0


# Internal classification codes -> taxonomy label + pedagogy template.
_TAXONOMY: dict[str, dict[str, str]] = {
    "broadcasting": {
        "label": "Broadcasting error",
        "concept": "Reduction dimensions and keepdim",
        "inspect": "the shape of your row-sum before you divide - print it.",
        "next_step": "Pass keepdim=True to the reduction so the sum is (N, 1) and broadcasts against (N, M).",
    },
    "shape_reduced": {
        "label": "Dimension-selection error",
        "concept": "Reduction dimensions and keepdim",
        "inspect": "the rank of your output vs the input - a dimension went missing.",
        "next_step": "Keep the reduced dimension (keepdim=True) so the output shape matches the input.",
    },
    "shape": {
        "label": "Shape error",
        "concept": "Tensor shapes",
        "inspect": "your output shape against the required shape.",
        "next_step": "Adjust the operation so the output shape matches what the challenge asks for.",
    },
    "value": {
        "label": "Numerical error",
        "concept": "Tensor operations",
        "inspect": "whether the operation you applied produces the correct values for the expected output.",
        "next_step": "Check your operation parameters (dimension, axis) and compare a small example by hand.",
    },
    "dtype": {
        "label": "Dtype mismatch",
        "concept": "Tensor dtypes",
        "inspect": "the dtype of your output tensor.",
        "next_step": "Cast or create the tensor with the correct dtype.",
    },
    "forbidden_api": {
        "label": "Forbidden API usage",
        "concept": "Vectorization",
        "inspect": "the constraint: this challenge forbids Python loops.",
        "next_step": "Replace the loop with a single tensor operation.",
    },
    "type": {
        "label": "Type error",
        "concept": "Return values",
        "inspect": "what your function returns - it must return a torch.Tensor.",
        "next_step": "Make sure you return the computed tensor.",
    },
    "timeout": {
        "label": "Timeout",
        "concept": "Performance",
        "inspect": "whether your code has an unbounded loop or very large allocation.",
        "next_step": "Use a vectorized operation instead of iterating.",
    },
    "runtime": {
        "label": "Runtime exception",
        "concept": "Execution",
        "inspect": "the exception message below.",
        "next_step": "Fix the error it points to, then run again.",
    },
    "definition": {
        "label": "Runtime exception",
        "concept": "Function definition",
        "inspect": "whether the required function is defined with the exact name.",
        "next_step": "Define the function with the expected name and signature.",
    },
}


def _classify(check: dict[str, Any]) -> tuple[str, list[str]]:
    """Return (taxonomy code, observation lines) for a failing check."""
    category = check.get("category")
    detail = check.get("detail") or ""
    err = check.get("error") or {}
    msg = (err.get("message") or "").lower()
    observed = check.get("observed") or {}
    expected = check.get("expected") or {}
    obs: list[str] = []

    if category == "runtime_exception":
        if "size of tensor" in msg or "must match" in msg or "broadcast" in msg:
            obs.append(f"Running your function raised: {err.get('type')}: {err.get('message')}")
            obs.append("That happens when a reduced (N,) vector is divided into an (N, M) tensor.")
            return "broadcasting", obs
        obs.append(f"Your function raised {err.get('type')}: {err.get('message')}")
        return "runtime", obs

    if category == "shape":
        o, e = observed.get("shape"), expected.get("shape")
        if o is not None and e is not None:
            obs.append(f"Output shape was {tuple(o)}, but the challenge requires {tuple(e)}.")
            if len(o) < len(e):
                return "shape_reduced", obs
        return "shape", obs

    if category == "value":
        if "row_sums" in observed:
            obs.append(f"Shapes are fine, but the row sums came out as {observed['row_sums']} instead of all 1s.")
        else:
            obs.append("The output shape is correct, but the values are wrong.")
        return "value", obs

    if category == "dtype":
        o, e = observed.get("dtype"), expected.get("dtype")
        if o and e:
            obs.append(f"Output dtype is {o}, but the challenge requires {e}.")
        return "dtype", obs

    if category == "forbidden_api":
        obs.append(detail)
        return "forbidden_api", obs

    if category == "type":
        obs.append(detail)
        return "type", obs

    if category == "timeout":
        obs.append(detail)
        return "timeout", obs

    obs.append(detail or "A check failed.")
    return "runtime", obs


def _sanitize(check: dict[str, Any]) -> CheckResult:
    """Hidden checks expose only name + pass/fail + a safe one-line detail."""
    visibility = check.get("visibility", "visible")
    detail = check.get("detail")
    if visibility == "hidden" and not check.get("passed"):
        detail = "hidden check failed"
    return CheckResult(
        name=check.get("name", "?"),
        visibility=visibility,
        passed=bool(check.get("passed")),
        detail=detail,
        category=check.get("category"),
    )


def build_feedback(runner_result: dict[str, Any], challenge: Challenge, *, drill_id: str | None = None) -> Feedback:
    capsule = challenge.concept_capsules[0] if challenge.concept_capsules else None
    drill = drill_id

    # Load / definition failure.
    if not runner_result.get("loaded", False):
        err = runner_result.get("load_error") or {}
        code = "definition" if err.get("type") in {"NameError"} else "runtime"
        tmpl = _TAXONOMY[code]
        return Feedback(
            status="error",
            summary=f"{tmpl['label']}: {err.get('type')}: {err.get('message')}",
            mistake_category=tmpl["label"],
            observations=[f"{err.get('type')}: {err.get('message')}"],
            concept=tmpl["concept"], inspect=tmpl["inspect"], next_step=tmpl["next_step"],
            capsule_suggestion=capsule, drill_suggestion=drill,
            total_count=0, passed_count=0,
        )

    checks: list[dict[str, Any]] = runner_result.get("checks", [])
    results = [_sanitize(c) for c in checks]
    passed = sum(1 for c in checks if c.get("passed"))
    total = len(checks)
    visible = [r for r in results if r.visibility == "visible"]
    hidden = [r for r in results if r.visibility == "hidden"]

    if total > 0 and passed == total:
        return Feedback(
            status="passed",
            summary=f"All {total} checks passed. Nicely done.",
            visible_results=visible, hidden_results=hidden,
            passed_count=passed, total_count=total,
        )

    failing = next((c for c in checks if not c.get("passed")), None)
    if failing is None:
        return Feedback(status="failed", summary="Some checks did not pass.",
                        visible_results=visible, hidden_results=hidden,
                        passed_count=passed, total_count=total)

    code, observations = _classify(failing)
    tmpl = _TAXONOMY[code]
    where = "a hidden check" if failing.get("visibility") == "hidden" else f"check '{failing.get('name')}'"
    return Feedback(
        status="failed",
        summary=f"{passed}/{total} checks passed. {tmpl['label']} in {where}.",
        mistake_category=tmpl["label"],
        observations=observations,
        concept=tmpl["concept"],
        inspect=tmpl["inspect"],
        next_step=tmpl["next_step"],
        capsule_suggestion=capsule,
        drill_suggestion=drill,
        visible_results=visible,
        hidden_results=hidden,
        passed_count=passed,
        total_count=total,
    )
