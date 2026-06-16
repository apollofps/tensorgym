"""Shared test fixtures. Adds the API package to the path and starts each test
session from a clean SQLite database."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
API_DIR = REPO_ROOT / "apps" / "api"
for p in (str(API_DIR),):
    if p not in sys.path:
        sys.path.insert(0, p)

from app.config import DB_PATH, EXECUTOR_IMAGE  # noqa: E402
from app.executor_service import docker_ready  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def _clean_db() -> None:
    if DB_PATH.exists():
        DB_PATH.unlink()
    yield


@pytest.fixture(scope="session")
def docker_or_skip() -> None:
    status = docker_ready()
    if not status["daemon"]:
        pytest.skip("Docker daemon not available")
    if not status["image"]:
        pytest.skip(f"executor image {EXECUTOR_IMAGE} not built")


@pytest.fixture
def client():
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as c:
        yield c
