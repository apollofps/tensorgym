# TensorGym

Challenge-first, just-in-time, **documentation-grounded** way to learn PyTorch by
solving, running, and debugging real tensor code — not by reading long chapters.

> **Status: Milestone 1 — one complete, tested vertical slice.**
> The full challenge library (30 challenges / 15 capsules) is intentionally **not**
> generated yet. See `RESEARCH_MANIFEST.md` and `docs/` for scope and decisions.

## What the slice does

A single beginner challenge, **Normalize Each Row**, drives the entire learning
loop end to end:

```
challenge → "I don't know this yet" → Concept Capsule → executable example →
tiny warm-up → return (code preserved) → run & submit (visible + hidden tests) →
targeted feedback → progress + mastery update
```

All learner code executes in a **hardened, network-isolated Docker sandbox**
pinned to **PyTorch 2.12.0 (CPU) / Python 3.12** — never in the API process.

## Pinned environment (verified 2026-06-15)

| | |
|---|---|
| PyTorch | **2.12.0** (latest stable, released 2026-05-13) |
| Python (sandbox) | 3.12 |
| Device | CPU only (labeled CPU-verified) |
| Image | `tensorgym-executor:py312-torch2.12.0-cpu` |

Version and documentation facts are verified from official sources in
[`RESEARCH_MANIFEST.md`](RESEARCH_MANIFEST.md).

## Prerequisites
- Docker (daemon running)
- Python 3.11+ on the host (for the API/tests; the host never needs torch)

## Quickstart

```bash
make setup            # venv + host deps (no torch on the host)
make build-executor   # build the pinned PyTorch 2.12.0 CPU sandbox image
make test             # full suite, runs real code in Docker (~22s)
make run              # serve API + UI at http://127.0.0.1:8000
```

Open http://127.0.0.1:8000 and walk the flow: click **I don't know this yet**,
read the capsule, run the example, do the warm-up, return (your code is preserved),
then **Submit**.

## Tests

`make test` runs 19 tests:
- `test_content_verification.py` — reference solution, capsule example, and warm-up
  all run in the **pinned Docker env**; published items are `verified` with sources.
- `test_executor.py` — sandbox security (no network, timeout reaping, AST loop
  ban) + structured result protocol + tensor metadata.
- `test_feedback.py` — deterministic mistake-taxonomy mapping (no Docker).
- `test_flow_e2e.py` — the whole slice through the API + Docker, including
  code-preservation and durable persistence across a restart.

`make test-fast` runs only the non-Docker unit tests.

## Layout

```
apps/web/            minimal static UI (textarea editor + fetch) — drives the flow
apps/api/app/        FastAPI: content loader, feedback engine, progress, endpoints
services/executor/   Dockerfile + runner.py (in-container) + docker_executor.py (host)
content/             challenge + capsule + warm-up + sources (data, with verification)
tests/               unit + executor + content-verification + e2e
docs/                ARCHITECTURE.md · THREAT_MODEL.md · LEARNER_FLOWS.md
RESEARCH_MANIFEST.md research provenance and pinned-env decisions
```

## Security model
Learner code runs only in Docker with `--network none`, memory/CPU/pids caps, a
read-only root, dropped capabilities, `no-new-privileges`, a non-root user, and a
wall-clock timeout. The Docker socket is never exposed to learner code. Full
detail and the tests that assert it: [`docs/THREAT_MODEL.md`](docs/THREAT_MODEL.md).

## Not in this milestone (deferred, by design)
Monaco/Next.js UI · the full content library · diagnostic · content-audit command ·
visualizations beyond the tensor inspector · mini-projects · GPU / `torch.compile`
/ vLLM roadmap items. These have extension points but are out of scope until the
slice is reviewed.
