"""Host-side sandbox executor.

Runs untrusted learner code inside a hardened, network-isolated Docker container
and returns a structured result. This module runs in the executor service; the API
calls it but NEVER executes learner code itself.

Security flags applied to every run (see docs/THREAT_MODEL.md):
  --network none              no network access
  --memory / --memory-swap    hard memory cap (no swap escape)
  --cpus                      CPU cap
  --pids-limit                fork-bomb protection
  --read-only + --tmpfs       read-only root; only a small writable /tmp
  --cap-drop ALL              drop all Linux capabilities
  --security-opt no-new-privileges
  read-only bind mount of the ephemeral job dir; Docker socket is never mounted
A wall-clock timeout plus `docker kill` guarantees runaway containers are reaped.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

DEFAULT_IMAGE = "tensorgym-executor:py312-torch2.12.0-cpu"
RESULT_SENTINEL = "__TENSORGYM_RESULT__"


@dataclass
class ExecResult:
    ok: bool                       # runner produced a parseable result
    result: dict[str, Any] | None  # parsed structured result from runner.py
    timed_out: bool
    exit_code: int | None
    duration_s: float
    stdout: str = ""
    stderr: str = ""
    error: str | None = None       # host-side error (e.g. image missing)


class Executor(Protocol):
    """Backend-agnostic execution interface (Docker now; GPU/remote/k8s later)."""

    def run(self, job: dict[str, Any], *, timeout_s: int = 15) -> ExecResult: ...


@dataclass
class DockerExecutor:
    image: str = DEFAULT_IMAGE
    memory: str = "512m"
    cpus: str = "1"
    pids_limit: int = 128
    tmpfs_size: str = "64m"
    docker_bin: str = "docker"
    # Ephemeral job dirs are created here. On macOS this must be under a path
    # Docker Desktop shares (e.g. inside the repo) - the system temp dir may not be.
    job_base: str | None = None
    extra_run_args: list[str] = field(default_factory=list)

    # -- introspection -------------------------------------------------------
    def image_exists(self) -> bool:
        proc = subprocess.run(
            [self.docker_bin, "image", "inspect", self.image],
            capture_output=True, text=True,
        )
        return proc.returncode == 0

    def daemon_available(self) -> bool:
        proc = subprocess.run([self.docker_bin, "info"], capture_output=True, text=True)
        return proc.returncode == 0

    # -- execution -----------------------------------------------------------
    def run(self, job: dict[str, Any], *, timeout_s: int = 15) -> ExecResult:
        import time

        if not self.image_exists():
            return ExecResult(False, None, False, None, 0.0,
                              error=f"executor image '{self.image}' not found; build it first")

        # in-container watchdog slightly below the host wall-clock timeout
        job = {**job, "soft_timeout_s": min(job.get("soft_timeout_s", timeout_s - 2), timeout_s - 1)}

        if self.job_base:
            Path(self.job_base).mkdir(parents=True, exist_ok=True)
        workdir = Path(tempfile.mkdtemp(prefix="tg-job-", dir=self.job_base))
        name = f"tg-{uuid.uuid4().hex[:12]}"
        started = time.monotonic()
        try:
            (workdir / "job.json").write_text(json.dumps(job), encoding="utf-8")

            cmd = [
                self.docker_bin, "run", "--rm", "--name", name,
                "--network", "none",
                "--memory", self.memory, "--memory-swap", self.memory,
                "--cpus", self.cpus,
                "--pids-limit", str(self.pids_limit),
                "--read-only",
                "--tmpfs", f"/tmp:rw,size={self.tmpfs_size}",
                "--cap-drop", "ALL",
                "--security-opt", "no-new-privileges",
                "-v", f"{workdir}:/work:ro",
                "--workdir", "/work",
                *self.extra_run_args,
                self.image,
                "/work/job.json",
            ]

            try:
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
            except subprocess.TimeoutExpired as exc:
                subprocess.run([self.docker_bin, "kill", name], capture_output=True, text=True)
                return ExecResult(
                    False, None, True, None, time.monotonic() - started,
                    stdout=_as_text(exc.stdout), stderr=_as_text(exc.stderr),
                    error=f"execution exceeded {timeout_s}s wall-clock limit",
                )

            duration = time.monotonic() - started
            parsed = _extract_result(proc.stdout)
            return ExecResult(
                ok=parsed is not None,
                result=parsed,
                timed_out=False,
                exit_code=proc.returncode,
                duration_s=duration,
                stdout=proc.stdout[-8000:],
                stderr=proc.stderr[-8000:],
                error=None if parsed is not None else "runner produced no parseable result",
            )
        finally:
            shutil.rmtree(workdir, ignore_errors=True)


def _as_text(value: Any) -> str:  # noqa: ANN401
    if value is None:
        return ""
    return value if isinstance(value, str) else value.decode("utf-8", "replace")


def _extract_result(stdout: str) -> dict[str, Any] | None:
    """Pull the single sentinel-prefixed JSON line out of stdout."""
    for line in reversed(stdout.splitlines()):
        if line.startswith(RESULT_SENTINEL):
            try:
                return json.loads(line[len(RESULT_SENTINEL):])
            except json.JSONDecodeError:
                return None
    return None
