"""SQLite persistence via SQLAlchemy 2.0."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from .config import DB_PATH


class Base(DeclarativeBase):
    pass


_engine = create_engine(f"sqlite:///{DB_PATH}", future=True)
SessionLocal = sessionmaker(bind=_engine, autoflush=False, expire_on_commit=False, future=True)


def init_db() -> None:
    from . import models  # noqa: F401 - register tables

    Base.metadata.create_all(_engine)


@contextmanager
def session_scope() -> Iterator:
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
