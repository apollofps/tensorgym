# Learner Flows (MVP slice)

The implemented slice realizes the spec's prerequisite-recovery loop end to end
for one beginner challenge: **Normalize Each Row** (`tensor-row-normalize-001`).

## Main flow

```
Open challenge ─► readiness panel shows required prereq: reduction-dimensions [?]
   │
   ├─(a) attempt now ─► Run / Submit
   │
   └─(b) "I don't know this yet"
           │ editor code saved to workspace (preserved)
           ▼
        Concept Capsule: reduction-dimensions-keepdim
           │  concise sourced explanation + shape example + verification status
           ▼
        Executable example  ─► sandbox (script mode) ─► stdout (shows (N,) vs (N,1))
           ▼
        Tiny warm-up: row-sums-keepdim-warmup
           │  Run ─► visible checks ─► "completed" unlocks Return
           ▼
        Return to challenge  ─► workspace code restored (preserved), hint level kept
           ▼
        Run (visible)  /  Submit (visible + hidden)
           ▼
        Feedback (deterministic taxonomy):
           - pass  ─► mastery↑ (reduction-dimensions, broadcasting) + ONE next action
           - shape/broadcasting error ─► targeted four-part feedback + capsule link
           ▼
        Progress persisted (SQLite) ─► resume after restart
```

## What is preserved across the detour and across restarts
| State | Where | Restored by |
|---|---|---|
| In-progress editor code | `workspace.editor_code` | `GET /api/challenges/{id}` |
| Hint level reached | `workspace.hint_level` | same |
| Test history (run/submit) | `attempt` rows | `GET /api/progress` |
| Concept mastery | `mastery` rows | `GET /api/progress` |
| Evidence (capsule opened, warm-up done, idk) | `event` rows | — |

## Progressive help (Learn panel)
Hint levels 0–4 are separate, server-tracked actions:
- L0 concept reminder · L1 directional · L2 structured · L3 partial code ·
  L4 full walkthrough. The number of hints revealed reduces mastery credit
  (`1.0 − 0.1·hints`, floored at 0.6).

## Feedback shape (spec's four parts)
1. **What was observed** (e.g. "running your function raised RuntimeError: size of
   tensor a (3) must match b (2)").
2. **Concept involved** ("Reduction dimensions and keepdim").
3. **What to inspect** ("the shape of your row-sum before you divide").
4. **Smallest next step** ("pass `keepdim=True` so the sum is (N,1) and broadcasts").

Hidden-check internals are never leaked: a failed hidden check reports only
`hidden check failed`.
