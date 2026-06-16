# Code-Execution Threat Model

## Scope
Learners submit arbitrary Python. Treat every submission as hostile.

- **Assets:** the host machine, the API process, other learners' data, the host
  network, the Docker daemon.
- **Adversary:** the learner-submitted code string.
- **Trust boundary:** the Docker container. Nothing the learner writes executes
  outside it. The API process is on the trusted side and must stay that way.

## Core invariants
1. The API **never** `exec`/`eval`s a submission and **never** imports torch.
   Only `runner.py`, inside the container, executes learner code.
2. The Docker **socket is never mounted** into the container — learner code cannot
   reach the daemon.
3. Containers are **ephemeral** and **`--rm`** — one per run, no shared state.

## Controls (every run; enforced in `services/executor/docker_executor.py`)

| Threat | Control | Flag |
|---|---|---|
| Network exfiltration / pivot | no network | `--network none` |
| CPU exhaustion | CPU cap | `--cpus 1` |
| Memory bomb | hard cap, no swap escape | `--memory 512m --memory-swap 512m` |
| Fork bomb | process cap | `--pids-limit 128` |
| Filesystem tampering / persistence | read-only root; only small writable tmp | `--read-only` + `--tmpfs /tmp:rw,size=64m` |
| Privilege escalation | drop caps; no setuid escalation; non-root | `--cap-drop ALL` + `--security-opt no-new-privileges` + image `USER runner` |
| Infinite loop / hang | host wall-clock timeout → `docker kill`, plus in-container `SIGALRM` watchdog | `subprocess timeout` + `signal.alarm` |
| Output as attack vector | single sentinel-prefixed JSON line, size-capped (8 KB), parsed defensively | `runner.py` `_clip`, `_extract_result` |
| Daemon abuse | socket never mounted | (absence of `-v /var/run/docker.sock`) |
| Host path exposure | only an ephemeral job dir, mounted **read-only** | `-v {jobdir}:/work:ro` |

## Tests that assert these properties
`tests/test_executor.py`:
- `test_network_is_disabled` — an outbound `connect()` fails inside the box.
- `test_timeout_is_enforced` — `while True: pass` is reaped (watchdog or wall-clock).
- `test_no_loops_static_check` — constraint enforcement (AST scan) before execution.
- `test_structured_pass_and_metadata` — structured protocol + tensor metadata.

## Residual risk / not-yet-done (later milestones)
- No seccomp/AppArmor profile yet (defense-in-depth beyond `--cap-drop ALL`).
- No user namespace remapping configured at the daemon level.
- CPU-only; GPU execution will need its own isolation review.
- Output size is capped but not rate-limited across many rapid submissions.
These are acceptable for a single-user local MVP and are tracked for hardening.
