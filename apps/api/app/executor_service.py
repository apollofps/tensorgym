"""Thin facade the API uses to reach the sandboxed executor.

The API never runs learner code itself; it only builds a job and hands it to the
Docker-backed executor (imported from services/executor via sys.path in config).
"""

from __future__ import annotations

from typing import Any

from .config import EXECUTOR_IMAGE, VAR_DIR

# docker_executor is on sys.path thanks to config.EXECUTOR_DIR insertion.
from docker_executor import DockerExecutor, ExecResult  # type: ignore  # noqa: E402

# Job dirs live under the repo (Docker Desktop shares /Users by default).
_executor = DockerExecutor(image=EXECUTOR_IMAGE, job_base=str(VAR_DIR / "jobs"))


def docker_ready() -> dict[str, bool]:
    return {"daemon": _executor.daemon_available(), "image": _executor.image_exists()}


def run_function_checks(
    function_name: str,
    code: str,
    checks: list[dict[str, Any]],
    metadata_input: Any | None = None,
    timeout_s: int = 15,
) -> ExecResult:
    job = {
        "mode": "checks",
        "submission_code": code,
        "function_name": function_name,
        "checks": checks,
        "metadata_input": metadata_input,
    }
    return _executor.run(job, timeout_s=timeout_s)


def run_script(code: str, timeout_s: int = 15) -> ExecResult:
    return _executor.run({"mode": "script", "submission_code": code}, timeout_s=timeout_s)
