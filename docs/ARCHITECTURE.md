# TensorGym Architecture (MVP slice)

## Principles
- **Challenge-first, just-in-time, documentation-grounded.**
- **The API never runs learner code and never imports torch.** All execution
  happens in a hardened Docker sandbox reached through an executor service.
- **Content is data, not code** — curriculum lives under `content/` and is loaded,
  not hard-coded.

## Tiers

```
┌──────────────┐   HTTP/JSON    ┌──────────────────────────┐  docker run (hardened)  ┌────────────────────┐
│  apps/web    │ ─────────────► │        apps/api          │ ──────────────────────► │ services/executor  │
│ static UI    │ ◄───────────── │  FastAPI + SQLite        │ ◄────────────────────── │  Docker sandbox    │
│ (textarea)   │   JSON         │  feedback + progress     │   structured JSON       │  PyTorch 2.12.0 CPU │
└──────────────┘                └──────────────────────────┘                         └────────────────────┘
                                          │
                                          ▼
                                  content/ (YAML + py)
```

### apps/web
Single static `index.html` (vanilla JS, `fetch`). Intentionally **not** Monaco /
Next.js yet — the spec says do not start with a design system. It drives the full
flow and preserves nothing critical client-side (the server is the source of truth
for preserved code, hints, history, mastery).

### apps/api (FastAPI, strict-typed)
- `config.py` — paths + pinned constants; puts the executor on `sys.path`.
- `db.py` / `models.py` — SQLAlchemy 2.0 + SQLite. Tables: `workspace`
  (preserved code + hint level), `attempt` (test history), `mastery`, `event`.
- `schemas.py` — Pydantic v2 content + request models.
- `content.py` — loads challenges/capsules/warm-ups/sources from `content/`.
- `feedback.py` — **deterministic** mistake taxonomy → four-part feedback.
- `progress.py` — mastery scoring + the single next-action rule + snapshots.
- `executor_service.py` — the only thing that talks to the sandbox.
- `main.py` — endpoints (below).

The API **must not** `exec`/`eval` learner code or import torch. This is the core
isolation invariant.

### services/executor
- `Dockerfile` — `python:3.12-slim` + `torch==2.12.0` (CPU index) + numpy + a
  non-root `runner` user. Pinned tag `tensorgym-executor:py312-torch2.12.0-cpu`.
- `runner.py` — runs **inside** the container. Loads the submission, runs
  declarative `kind`-based checks (or a script), reports facts + tensor metadata as
  one sentinel-prefixed JSON line. Has an in-container `SIGALRM` watchdog.
- `docker_executor.py` — host wrapper. Builds an ephemeral job dir, runs the
  container with the full hardening flag set (see `THREAT_MODEL.md`), enforces a
  wall-clock timeout with `docker kill`, parses the JSON result. Exposes an
  `Executor` Protocol so GPU / remote / k8s backends can be added later.

## API endpoints (slice)
| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | pinned versions + executor readiness |
| GET | `/api/challenges/{id}` | prompt, prereqs, preserved editor code, history |
| POST | `/api/challenges/{id}/workspace` | autosave editor code + hint level |
| POST | `/api/challenges/{id}/idk` | "I don't know this yet" → preserve + route to capsule |
| POST | `/api/challenges/{id}/hint?level=N` | progressive hint (levels 0–4) |
| GET | `/api/challenges/{id}/solution` | reference solution (full walkthrough) |
| POST | `/api/challenges/{id}/run` | visible checks + metadata + feedback (no credit) |
| POST | `/api/challenges/{id}/submit` | visible + hidden + feedback + mastery + next action |
| GET | `/api/capsules/{id}` | capsule + sources + verification + example code |
| POST | `/api/capsules/{id}/run-example` | run the executable example (script mode) |
| GET | `/api/warmups/{id}` | warm-up prompt + starter |
| POST | `/api/warmups/{id}/run` | run warm-up checks; mark complete |
| GET | `/api/progress` | mastery + workspace + attempts (resume after restart) |
| GET | `/` | the static UI |

## Result protocol (executor → API)
`runner.py` prints one line: `__TENSORGYM_RESULT__{json}`. The host extracts that
line so library chatter on stdout/stderr cannot corrupt the result. The JSON
carries per-check `{passed, category, detail, error, observed, expected}` plus
optional tensor `metadata`. Pedagogy (taxonomy labels, hints) is added by the API,
never the runner.

## Extension points (later milestones)
- `Executor` Protocol → local-GPU / remote-GPU / k8s backends.
- Content schemas already carry `sources` + `verification` for a content-audit tool.
- Monaco/Next.js UI can replace `apps/web` without touching the API contract.
