"""TensorGym API - the vertical slice.

Flow: challenge -> "I don't know this yet" -> capsule -> example -> warm-up ->
return (code preserved) -> run/submit (visible + hidden) -> targeted feedback ->
progress update.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import content, executor_service, progress
from .config import EXECUTOR_IMAGE, PINNED_PYTHON, PINNED_PYTORCH, WEB_DIR
from .db import init_db, session_scope
from .feedback import build_feedback
from .schemas import CodeBody, RunBody, ScriptBody, WorkspaceBody


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    init_db()
    yield


app = FastAPI(title="TensorGym API", version="0.1.0", lifespan=lifespan)


# --------------------------------------------------------------------------- #
# Meta
# --------------------------------------------------------------------------- #
@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "pinned": {"pytorch": PINNED_PYTORCH, "python": PINNED_PYTHON, "image": EXECUTOR_IMAGE},
        "executor": executor_service.docker_ready(),
    }


# --------------------------------------------------------------------------- #
# Challenge
# --------------------------------------------------------------------------- #
@app.get("/api/challenges")
def list_challenges() -> dict[str, Any]:
    from .config import CONTENT_DIR
    challenge_dir = CONTENT_DIR / "challenges"
    items = []
    with session_scope() as s:
        mastery = progress.get_mastery(s)
    for f in sorted(challenge_dir.glob("*.yaml")):
        try:
            ch = content.load_challenge(f.stem)
        except Exception:
            continue
        items.append({
            "id": ch.id, "title": ch.title, "difficulty": ch.difficulty,
            "type": ch.type, "estimated_minutes": ch.estimated_minutes,
            "concepts_tested": ch.concepts_tested,
        })
    return {"challenges": items}


def _prereq_status(mastery: dict[str, float], prereqs: list[str]) -> list[dict[str, Any]]:
    out = []
    for p in prereqs:
        score = mastery.get(p, 0.0)
        out.append({"id": p, "score": score, "status": "known" if score >= progress.MASTERY_FLOOR else "unknown"})
    return out


@app.get("/api/challenges/{challenge_id}")
def get_challenge(challenge_id: str) -> dict[str, Any]:
    try:
        ch = content.load_challenge(challenge_id)
    except FileNotFoundError:
        raise HTTPException(404, f"challenge '{challenge_id}' not found")

    with session_scope() as s:
        mastery = progress.get_mastery(s)
        ws = progress.get_workspace(s, challenge_id)
        attempts = progress.get_attempts(s, challenge_id)
        editor_code = ws.editor_code if ws and ws.editor_code else ch.starter_code
        hint_level = ws.hint_level if ws else -1
        history = [{"kind": a.kind, "passed": a.passed, "hints_used": a.hints_used,
                    "summary": a.summary, "at": a.created_at.isoformat()} for a in attempts]

    return {
        "id": ch.id, "title": ch.title, "difficulty": ch.difficulty, "type": ch.type,
        "prompt": ch.prompt, "constraints": ch.constraints,
        "function_name": ch.function_name,
        "starter_code": ch.starter_code,
        "editor_code": editor_code,          # preserved code (or starter on first visit)
        "hint_level": hint_level,
        "estimated_minutes": ch.estimated_minutes,
        "concepts_tested": ch.concepts_tested,
        "prerequisites": _prereq_status(mastery, ch.required_prerequisites),
        "concept_capsules": ch.concept_capsules,
        "visible_tests": [
            {"name": t.name, "description": t.description, "input": t.input} for t in ch.visible_tests
        ],
        "hint_count": len(ch.hints),
        "verification": ch.verification.model_dump(),
        "attempts": history,
    }


@app.post("/api/challenges/{challenge_id}/workspace")
def save_workspace(challenge_id: str, body: WorkspaceBody) -> dict[str, Any]:
    with session_scope() as s:
        ws = progress.save_workspace(s, challenge_id, body.code, body.hint_level)
        return {"saved": True, "hint_level": ws.hint_level}


@app.post("/api/challenges/{challenge_id}/idk")
def i_dont_know(challenge_id: str, body: WorkspaceBody) -> dict[str, Any]:
    """'I don't know this yet': preserve code, route to the prerequisite capsule."""
    try:
        ch = content.load_challenge(challenge_id)
    except FileNotFoundError:
        raise HTTPException(404, "challenge not found")
    capsule_id = ch.concept_capsules[0] if ch.concept_capsules else None
    if not capsule_id:
        raise HTTPException(400, "no capsule linked to this challenge")
    cap = content.load_capsule(capsule_id)
    with session_scope() as s:
        progress.save_workspace(s, challenge_id, body.code, body.hint_level)
        progress.log_event(s, "idk_clicked", capsule_id)
    return {
        "preserved": True,
        "capsule": {"id": cap.id, "title": cap.title},
        "warmup_id": cap.warmup_challenge_id,
        "return_to": challenge_id,
        "flow": ["capsule", "example", "warmup", "return_to_challenge"],
    }


@app.post("/api/challenges/{challenge_id}/hint")
def get_hint(challenge_id: str, level: int) -> dict[str, Any]:
    try:
        ch = content.load_challenge(challenge_id)
    except FileNotFoundError:
        raise HTTPException(404, "challenge not found")
    hint = next((h for h in ch.hints if h.level == level), None)
    if hint is None:
        raise HTTPException(404, f"no hint at level {level}")
    with session_scope() as s:
        ws = progress.bump_hint_level(s, challenge_id, level)
        return {"level": hint.level, "text": hint.text, "hint_level": ws.hint_level,
                "max_level": max(h.level for h in ch.hints)}


@app.get("/api/challenges/{challenge_id}/solution")
def get_solution(challenge_id: str) -> dict[str, Any]:
    try:
        ch = content.load_challenge(challenge_id)
    except FileNotFoundError:
        raise HTTPException(404, "challenge not found")
    return {"solution_code": content.load_solution_code(ch)}


def _visible_checks(ch) -> list[dict[str, Any]]:  # noqa: ANN001
    out = []
    for t in ch.visible_tests:
        d = t.model_dump()
        d["visibility"] = "visible"
        out.append(d)
    return out


def _execute_and_feedback(challenge_id: str, code: str, *, include_hidden: bool, kind: str) -> dict[str, Any]:
    ch = content.load_challenge(challenge_id)
    checks = _visible_checks(ch)
    if include_hidden:
        checks = checks + content.load_hidden_checks(ch)

    metadata_input = ch.visible_tests[0].input if ch.visible_tests else None
    exec_result = executor_service.run_function_checks(
        ch.function_name, code, checks, metadata_input=metadata_input
    )

    if not exec_result.ok or exec_result.result is None:
        # sandbox-level failure (timeout, image missing, unparseable)
        with session_scope() as s:
            progress.save_workspace(s, challenge_id, code)
            hints_used = max(0, (progress.get_workspace(s, challenge_id).hint_level) + 1)
            progress.record_attempt(s, challenge_id, kind, False, hints_used,
                                    exec_result.error or "execution failed")
        return {
            "ok": False,
            "execution_error": exec_result.error,
            "timed_out": exec_result.timed_out,
            "stderr": exec_result.stderr[-2000:],
            "duration_s": round(exec_result.duration_s, 3),
        }

    drill_id = None
    if ch.concept_capsules:
        try:
            cap = content.load_capsule(ch.concept_capsules[0])
            drill_id = cap.warmup_challenge_id
        except FileNotFoundError:
            pass
    fb = build_feedback(exec_result.result, ch, drill_id=drill_id)
    metadata = exec_result.result.get("metadata")

    with session_scope() as s:
        progress.save_workspace(s, challenge_id, code)
        ws = progress.get_workspace(s, challenge_id)
        hints_used = max(0, (ws.hint_level if ws else -1) + 1)
        progress.record_attempt(s, challenge_id, kind, fb.status == "passed", hints_used, fb.summary)

        mastery_update: dict[str, float] = {}
        if kind == "submit" and fb.status == "passed":
            mastery_update = progress.update_mastery_on_pass(s, ch.concepts_tested, hints_used)

        next_action = progress.recommend_next_action(fb, ch)
        snapshot = progress.progress_snapshot(s, challenge_id)

    return {
        "ok": True,
        "kind": kind,
        "feedback": fb.model_dump(),
        "metadata": metadata,
        "mastery_update": mastery_update,
        "next_action": next_action,
        "progress": snapshot,
        "duration_s": round(exec_result.duration_s, 3),
    }


@app.post("/api/challenges/{challenge_id}/run")
def run_challenge(challenge_id: str, body: CodeBody) -> dict[str, Any]:
    return _execute_and_feedback(challenge_id, body.code, include_hidden=False, kind="run")


@app.post("/api/challenges/{challenge_id}/submit")
def submit_challenge(challenge_id: str, body: RunBody) -> dict[str, Any]:
    return _execute_and_feedback(challenge_id, body.code, include_hidden=True, kind="submit")


# --------------------------------------------------------------------------- #
# Capsule + example
# --------------------------------------------------------------------------- #
@app.get("/api/capsules/{capsule_id}")
def get_capsule(capsule_id: str) -> dict[str, Any]:
    try:
        cap = content.load_capsule(capsule_id)
    except FileNotFoundError:
        raise HTTPException(404, "capsule not found")
    with session_scope() as s:
        progress.log_event(s, "capsule_opened", capsule_id)
    sources = content.resolve_sources(cap.sources)
    return {
        "id": cap.id, "title": cap.title, "estimated_minutes": cap.estimated_minutes,
        "supported_pytorch_version": cap.supported_pytorch_version,
        "summary": cap.summary,
        "learning_objectives": cap.learning_objectives,
        "common_mistakes": cap.common_mistakes,
        "example_code": content.load_example_code(cap),
        "warmup_id": cap.warmup_challenge_id,
        "verification": cap.verification.model_dump(),
        "sources": [{"id": s.id, "title": s.title, "url": s.url,
                     "documentation_version": s.documentation_version, "accessed_at": s.accessed_at}
                    for s in sources],
    }


@app.post("/api/capsules/{capsule_id}/run-example")
def run_example(capsule_id: str, body: ScriptBody | None = None) -> dict[str, Any]:
    try:
        cap = content.load_capsule(capsule_id)
    except FileNotFoundError:
        raise HTTPException(404, "capsule not found")
    code = body.code if body and body.code else content.load_example_code(cap)
    res = executor_service.run_script(code)
    if not res.ok or res.result is None:
        return {"ok": False, "execution_error": res.error, "timed_out": res.timed_out,
                "stderr": res.stderr[-2000:]}
    return {"ok": True, "stdout": res.result.get("stdout", ""), "stderr": res.result.get("stderr", ""),
            "error": res.result.get("error"), "duration_s": round(res.duration_s, 3)}


# --------------------------------------------------------------------------- #
# Warm-up
# --------------------------------------------------------------------------- #
@app.get("/api/warmups/{warmup_id}")
def get_warmup(warmup_id: str) -> dict[str, Any]:
    try:
        w = content.load_warmup(warmup_id)
    except FileNotFoundError:
        raise HTTPException(404, "warmup not found")
    return {
        "id": w.id, "title": w.title, "prompt": w.prompt, "starter_code": w.starter_code,
        "function_name": w.function_name, "constraints": w.constraints,
        "visible_tests": [{"name": t.name, "description": t.description} for t in w.visible_tests],
    }


@app.post("/api/warmups/{warmup_id}/run")
def run_warmup(warmup_id: str, body: CodeBody) -> dict[str, Any]:
    try:
        w = content.load_warmup(warmup_id)
    except FileNotFoundError:
        raise HTTPException(404, "warmup not found")
    checks = []
    for t in w.visible_tests:
        d = t.model_dump()
        d["visibility"] = "visible"
        checks.append(d)
    res = executor_service.run_function_checks(w.function_name, body.code, checks)
    if not res.ok or res.result is None:
        return {"ok": False, "execution_error": res.error, "timed_out": res.timed_out}
    checks_out = res.result.get("checks", [])
    passed = all(c.get("passed") for c in checks_out) and len(checks_out) > 0
    if passed:
        with session_scope() as s:
            progress.log_event(s, "warmup_completed", warmup_id)
    return {
        "ok": True,
        "completed": passed,
        "checks": [{"name": c.get("name"), "passed": c.get("passed"), "detail": c.get("detail")}
                   for c in checks_out],
    }


# --------------------------------------------------------------------------- #
# Progress (resume after restart)
# --------------------------------------------------------------------------- #
@app.get("/api/progress")
def get_progress(challenge_id: str | None = None) -> dict[str, Any]:
    with session_scope() as s:
        return progress.progress_snapshot(s, challenge_id)


# --------------------------------------------------------------------------- #
# Static web UI (mounted last so it doesn't shadow /api)
# --------------------------------------------------------------------------- #
@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


if (WEB_DIR / "index.html").exists():
    app.mount("/app", StaticFiles(directory=str(WEB_DIR), html=True), name="web")
