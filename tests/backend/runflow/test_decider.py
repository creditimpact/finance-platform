from __future__ import annotations

import json
from pathlib import Path

from backend.runflow.decider import decide_next, record_stage


def _load_runflow(tmp_root: Path, sid: str) -> dict:
    path = tmp_root / sid / "runflow.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_decide_next_requests_validation_when_pending(tmp_path):
    runs_root = tmp_path / "runs"
    sid = "S1"

    record_stage(
        sid,
        "merge",
        status="success",
        counts={"count": 2},
        empty_ok=False,
        runs_root=runs_root,
    )

    decision = decide_next(sid, runs_root=runs_root)
    assert decision["next"] == "run_validation"
    data = _load_runflow(runs_root, sid)
    assert data["run_state"] == "VALIDATING"


def test_decide_next_triggers_frontend_when_findings(tmp_path):
    runs_root = tmp_path / "runs"
    sid = "S2"

    record_stage(
        sid,
        "merge",
        status="success",
        counts={"count": 1},
        empty_ok=False,
        runs_root=runs_root,
    )
    record_stage(
        sid,
        "validation",
        status="success",
        counts={"findings_count": 3},
        empty_ok=False,
        runs_root=runs_root,
    )

    decision = decide_next(sid, runs_root=runs_root)
    assert decision["next"] == "gen_frontend_packs"
    data = _load_runflow(runs_root, sid)
    assert data["run_state"] == "VALIDATING"


def test_decide_next_completes_when_empty_ok(tmp_path):
    runs_root = tmp_path / "runs"
    sid = "S3"

    record_stage(
        sid,
        "merge",
        status="success",
        counts={"count": 0},
        empty_ok=True,
        runs_root=runs_root,
    )

    decision = decide_next(sid, runs_root=runs_root)
    assert decision["next"] == "complete_no_action"
    data = _load_runflow(runs_root, sid)
    assert data["run_state"] == "COMPLETE_NO_ACTION"


def test_decide_next_moves_to_await_after_frontend_success(tmp_path):
    runs_root = tmp_path / "runs"
    sid = "S4"

    record_stage(
        sid,
        "merge",
        status="success",
        counts={"count": 1},
        empty_ok=False,
        runs_root=runs_root,
    )
    record_stage(
        sid,
        "validation",
        status="success",
        counts={"findings_count": 2},
        empty_ok=False,
        runs_root=runs_root,
    )
    record_stage(
        sid,
        "frontend",
        status="success",
        counts={"packs_count": 1},
        empty_ok=False,
        runs_root=runs_root,
    )

    decision = decide_next(sid, runs_root=runs_root)
    assert decision["next"] == "await_input"
    data = _load_runflow(runs_root, sid)
    assert data["run_state"] == "AWAITING_CUSTOMER_INPUT"
