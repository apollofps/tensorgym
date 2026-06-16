"""End-to-end test of the full vertical slice through the API + real Docker.

Challenge -> "I don't know this yet" -> capsule -> example -> warm-up ->
return (code preserved) -> run (buggy -> targeted feedback) ->
submit (correct -> pass + mastery + next action) -> progress persists.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "apps" / "api"))

from app.config import DB_PATH  # noqa: E402

pytestmark = pytest.mark.usefixtures("docker_or_skip")

CID = "tensor-row-normalize-001"
GOOD = "import torch\n\ndef normalize_rows(x):\n    return x / x.sum(dim=1, keepdim=True)\n"
BAD = "import torch\n\ndef normalize_rows(x):\n    return x / x.sum(dim=1)\n"
WARM_GOOD = "import torch\n\ndef row_sums(x):\n    return x.sum(dim=1, keepdim=True)\n"
MARKER = "# my work in progress\n"


def test_full_slice(client):
    # 1. health
    h = client.get("/api/health").json()
    assert h["status"] == "ok" and h["pinned"]["pytorch"] == "2.12.0"
    assert h["executor"]["daemon"] and h["executor"]["image"]

    # 2. open challenge: starts from starter, prereq not yet known
    ch = client.get(f"/api/challenges/{CID}").json()
    assert ch["editor_code"] == ch["starter_code"]
    prereq = {p["id"]: p["status"] for p in ch["prerequisites"]}
    assert prereq["reduction-dimensions"] == "unknown"

    # 3. "I don't know this yet" -> preserve in-progress code, route to capsule
    idk = client.post(f"/api/challenges/{CID}/idk",
                      json={"code": MARKER + ch["starter_code"], "hint_level": -1}).json()
    assert idk["preserved"] is True
    cap_id = idk["capsule"]["id"]
    warm_id = idk["warmup_id"]
    assert cap_id == "reduction-dimensions-keepdim"

    # 4. capsule content is verified and sourced
    cap = client.get(f"/api/capsules/{cap_id}").json()
    assert cap["verification"]["status"] == "verified"
    assert any("pytorch.org" in s["url"] for s in cap["sources"])

    # 5. executable example runs in the sandbox and shows the keepdim shape
    ex = client.post(f"/api/capsules/{cap_id}/run-example", json={"code": cap["example_code"]}).json()
    assert ex["ok"] and "(2, 1)" in ex["stdout"]

    # 6-7. warm-up: correct solution completes it
    warm = client.post(f"/api/warmups/{warm_id}/run", json={"code": WARM_GOOD}).json()
    assert warm["ok"] and warm["completed"] is True

    # 8. RETURN to challenge: the in-progress code is preserved
    ch2 = client.get(f"/api/challenges/{CID}").json()
    assert ch2["editor_code"].startswith(MARKER)

    # 9. RUN the buggy (no-keepdim) version -> targeted broadcasting feedback, no credit
    run = client.post(f"/api/challenges/{CID}/run", json={"code": BAD}).json()
    assert run["ok"] is True
    fb = run["feedback"]
    assert fb["status"] == "failed"
    assert fb["mistake_category"] == "Broadcasting error"
    assert "keepdim=True" in fb["next_step"]
    assert run["next_action"]["action"] == "prerequisite_capsule"
    assert run["next_action"]["ref"] == "reduction-dimensions-keepdim"
    assert run["mastery_update"] == {}  # run never grants mastery

    # 10. SUBMIT the correct version -> visible + hidden pass, mastery + next action
    sub = client.post(f"/api/challenges/{CID}/submit", json={"code": GOOD}).json()
    assert sub["ok"] is True
    fbs = sub["feedback"]
    assert fbs["status"] == "passed"
    assert fbs["total_count"] == 6  # 2 visible + 4 hidden
    assert any(r["visibility"] == "hidden" for r in fbs["hidden_results"])
    assert sub["mastery_update"]["reduction-dimensions"] == 1.0
    assert sub["mastery_update"]["broadcasting"] == 1.0
    assert sub["next_action"]["action"] == "advance"
    assert sub["metadata"]["shape"] == [2, 3]

    # 11. progress reflects mastery + history (run + submit recorded)
    prog = client.get(f"/api/progress?challenge_id={CID}").json()
    assert prog["mastery"]["reduction-dimensions"] == 1.0
    kinds = [a["kind"] for a in prog["attempts"]]
    assert "run" in kinds and "submit" in kinds

    # 12. prereq now shows as known on reload
    ch3 = client.get(f"/api/challenges/{CID}").json()
    prereq3 = {p["id"]: p["status"] for p in ch3["prerequisites"]}
    assert prereq3["reduction-dimensions"] == "known"


def test_persistence_survives_restart(client):
    # The earlier flow committed to SQLite on disk; read it back with a fresh
    # connection (independent of the app process) to prove durable persistence.
    con = sqlite3.connect(DB_PATH)
    try:
        masteries = dict(con.execute("SELECT concept, score FROM mastery").fetchall())
        ws = con.execute("SELECT editor_code FROM workspace WHERE challenge_id=?", (CID,)).fetchone()
    finally:
        con.close()
    assert masteries.get("reduction-dimensions") == 1.0
    # workspace holds the most recently saved code (the correct submission).
    assert ws is not None and "keepdim=True" in ws[0]

    # A brand-new client (simulated restart) sees the same progress.
    fresh = client.get(f"/api/progress?challenge_id={CID}").json()
    assert fresh["mastery"]["reduction-dimensions"] == 1.0
