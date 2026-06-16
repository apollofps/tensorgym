"""Progress persistence, deterministic mastery scoring, and next-action rules."""

from __future__ import annotations

from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from .feedback import Feedback
from .models import DEFAULT_LEARNER, Attempt, Event, Mastery, Workspace
from .schemas import Challenge

MASTERY_FLOOR = 0.6


# ----- workspace (preserved code) -------------------------------------------
def get_workspace(session: Session, challenge_id: str) -> Workspace | None:
    return session.get(Workspace, (DEFAULT_LEARNER, challenge_id))


def save_workspace(session: Session, challenge_id: str, code: str, hint_level: int | None = None) -> Workspace:
    ws = get_workspace(session, challenge_id)
    if ws is None:
        ws = Workspace(learner_id=DEFAULT_LEARNER, challenge_id=challenge_id, editor_code=code,
                       hint_level=hint_level if hint_level is not None else -1)
        session.add(ws)
    else:
        ws.editor_code = code
        if hint_level is not None:
            ws.hint_level = max(ws.hint_level, hint_level)
    return ws


def bump_hint_level(session: Session, challenge_id: str, level: int) -> Workspace:
    ws = get_workspace(session, challenge_id)
    if ws is None:
        ws = Workspace(learner_id=DEFAULT_LEARNER, challenge_id=challenge_id, editor_code="", hint_level=level)
        session.add(ws)
    else:
        ws.hint_level = max(ws.hint_level, level)
    return ws


# ----- history & events ------------------------------------------------------
def record_attempt(session: Session, challenge_id: str, kind: str, passed: bool,
                   hints_used: int, summary: str) -> None:
    session.add(Attempt(learner_id=DEFAULT_LEARNER, challenge_id=challenge_id, kind=kind,
                        passed=passed, hints_used=hints_used, summary=summary))


def get_attempts(session: Session, challenge_id: str) -> list[Attempt]:
    stmt = select(Attempt).where(
        Attempt.learner_id == DEFAULT_LEARNER, Attempt.challenge_id == challenge_id
    ).order_by(Attempt.created_at)
    return list(session.scalars(stmt))


def log_event(session: Session, kind: str, ref: str = "") -> None:
    session.add(Event(learner_id=DEFAULT_LEARNER, kind=kind, ref=ref))


# ----- mastery ---------------------------------------------------------------
def update_mastery_on_pass(session: Session, concepts: list[str], hints_used: int) -> dict[str, float]:
    """Deterministic: full credit with no hints; -0.1 per hint, floored at 0.6."""
    candidate = max(MASTERY_FLOOR, round(1.0 - 0.1 * hints_used, 2))
    updated: dict[str, float] = {}
    for concept in concepts:
        row = session.get(Mastery, (DEFAULT_LEARNER, concept))
        if row is None:
            row = Mastery(learner_id=DEFAULT_LEARNER, concept=concept, score=candidate)
            session.add(row)
        else:
            row.score = max(row.score, candidate)
        updated[concept] = row.score
    return updated


def get_mastery(session: Session) -> dict[str, float]:
    rows = session.scalars(select(Mastery).where(Mastery.learner_id == DEFAULT_LEARNER))
    return {r.concept: round(r.score, 2) for r in rows}


# ----- next action (exactly one) --------------------------------------------
def recommend_next_action(feedback: Feedback, challenge: Challenge) -> dict[str, Any]:
    if feedback.status == "passed":
        return {
            "action": "advance",
            "title": "Advance to a harder challenge",
            "detail": "You normalized rows correctly. Next: a challenge that builds on broadcasting.",
            "ref": None,
        }
    if feedback.status == "error":
        return {
            "action": "revisit",
            "title": "Fix and run again",
            "detail": feedback.next_step or "Resolve the error, then run again.",
            "ref": None,
        }
    if feedback.capsule_suggestion:
        return {
            "action": "prerequisite_capsule",
            "title": f"Review the capsule: {feedback.concept}",
            "detail": feedback.next_step or "Study the prerequisite, then return.",
            "ref": feedback.capsule_suggestion,
        }
    return {
        "action": "drill",
        "title": "Try a targeted drill",
        "detail": feedback.next_step or "Practice the underlying step.",
        "ref": feedback.drill_suggestion,
    }


# ----- snapshot (resume after restart) --------------------------------------
def progress_snapshot(session: Session, challenge_id: str | None = None) -> dict[str, Any]:
    snap: dict[str, Any] = {"mastery": get_mastery(session)}
    if challenge_id:
        ws = get_workspace(session, challenge_id)
        snap["workspace"] = (
            {"challenge_id": challenge_id, "editor_code": ws.editor_code, "hint_level": ws.hint_level}
            if ws else None
        )
        snap["attempts"] = [
            {"kind": a.kind, "passed": a.passed, "hints_used": a.hints_used,
             "summary": a.summary, "at": a.created_at.isoformat()}
            for a in get_attempts(session, challenge_id)
        ]
    return snap
