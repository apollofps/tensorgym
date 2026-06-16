"""Load curriculum content from the content/ tree (data, not code)."""

from __future__ import annotations

import importlib.util
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from .config import CONTENT_DIR
from .schemas import Capsule, Challenge, Source, Warmup

CHALLENGE_DIR = CONTENT_DIR / "challenges"
CAPSULE_DIR = CONTENT_DIR / "capsules"
WARMUP_DIR = CONTENT_DIR / "capsules" / "warmups"
SOURCES_DIR = CONTENT_DIR / "sources"


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"content not found: {path}")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=None)
def load_challenge(challenge_id: str) -> Challenge:
    return Challenge.model_validate(_read_yaml(CHALLENGE_DIR / f"{challenge_id}.yaml"))


@lru_cache(maxsize=None)
def load_capsule(capsule_id: str) -> Capsule:
    return Capsule.model_validate(_read_yaml(CAPSULE_DIR / f"{capsule_id}.yaml"))


@lru_cache(maxsize=None)
def load_warmup(warmup_id: str) -> Warmup:
    return Warmup.model_validate(_read_yaml(WARMUP_DIR / f"{warmup_id}.yaml"))


@lru_cache(maxsize=None)
def load_sources() -> dict[str, Source]:
    registry: dict[str, Source] = {}
    for path in sorted(SOURCES_DIR.glob("*.yaml")):
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if data and "sources" in data:
            for s in data["sources"]:
                registry[s["id"]] = Source.model_validate(s)
    return registry


def resolve_sources(ids: list[str]) -> list[Source]:
    registry = load_sources()
    return [registry[i] for i in ids if i in registry]


def load_solution_code(challenge: Challenge) -> str:
    if not challenge.solution_file:
        return ""
    return (CHALLENGE_DIR / challenge.solution_file).read_text(encoding="utf-8")


def load_example_code(capsule: Capsule) -> str:
    if not capsule.interactive_example:
        return ""
    return (CAPSULE_DIR / capsule.interactive_example).read_text(encoding="utf-8")


def load_hidden_checks(challenge: Challenge) -> list[dict[str, Any]]:
    """Import the hidden test file (trusted authored content) and read CHECKS."""
    if not challenge.hidden_test_file:
        return []
    path = CHALLENGE_DIR / challenge.hidden_test_file
    spec = importlib.util.spec_from_file_location(f"hidden_{challenge.id}", path)
    if spec is None or spec.loader is None:
        return []
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return list(getattr(module, "CHECKS", []))
