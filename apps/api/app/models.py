"""ORM models. Single-user local MVP: learner_id defaults to 'local'."""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from .db import Base

DEFAULT_LEARNER = "local"


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Workspace(Base):
    """Preserved editor state per challenge - survives the capsule detour and restarts."""

    __tablename__ = "workspace"

    learner_id: Mapped[str] = mapped_column(String, primary_key=True, default=DEFAULT_LEARNER)
    challenge_id: Mapped[str] = mapped_column(String, primary_key=True)
    editor_code: Mapped[str] = mapped_column(Text, default="")
    hint_level: Mapped[int] = mapped_column(Integer, default=-1)  # -1 = no hint revealed yet
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=_now, onupdate=_now)


class Attempt(Base):
    """Test history: every Run and Submit."""

    __tablename__ = "attempt"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    learner_id: Mapped[str] = mapped_column(String, default=DEFAULT_LEARNER, index=True)
    challenge_id: Mapped[str] = mapped_column(String, index=True)
    kind: Mapped[str] = mapped_column(String)  # "run" | "submit"
    passed: Mapped[bool] = mapped_column(default=False)
    hints_used: Mapped[int] = mapped_column(Integer, default=0)
    summary: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)


class Mastery(Base):
    """Per-concept mastery score in [0, 1]."""

    __tablename__ = "mastery"

    learner_id: Mapped[str] = mapped_column(String, primary_key=True, default=DEFAULT_LEARNER)
    concept: Mapped[str] = mapped_column(String, primary_key=True)
    score: Mapped[float] = mapped_column(Float, default=0.0)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=_now, onupdate=_now)


class Event(Base):
    """Evidence log: capsule_opened, warmup_completed, idk_clicked, etc."""

    __tablename__ = "event"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    learner_id: Mapped[str] = mapped_column(String, default=DEFAULT_LEARNER, index=True)
    kind: Mapped[str] = mapped_column(String, index=True)
    ref: Mapped[str] = mapped_column(String, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)
