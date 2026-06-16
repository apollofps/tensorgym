"""Paths and pinned-environment constants for the TensorGym API."""

from __future__ import annotations

import sys
from pathlib import Path

# repo root is four levels up: apps/api/app/config.py -> repo
REPO_ROOT = Path(__file__).resolve().parents[3]
CONTENT_DIR = REPO_ROOT / "content"
EXECUTOR_DIR = REPO_ROOT / "services" / "executor"
WEB_DIR = REPO_ROOT / "apps" / "web"
VAR_DIR = REPO_ROOT / "var"
DB_PATH = VAR_DIR / "tensorgym.db"

# Pinned, verified environment (see RESEARCH_MANIFEST.md).
PINNED_PYTORCH = "2.12.0"
PINNED_PYTHON = "3.12"
EXECUTOR_IMAGE = "tensorgym-executor:py312-torch2.12.0-cpu"

# Make the executor service importable without packaging it.
if str(EXECUTOR_DIR) not in sys.path:
    sys.path.insert(0, str(EXECUTOR_DIR))

VAR_DIR.mkdir(parents=True, exist_ok=True)
