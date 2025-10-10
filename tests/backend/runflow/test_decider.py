from __future__ import annotations

import json
from pathlib import Path

from backend.runflow.decider import decide_next, record_stage


def _load_runflow(tmp_root: Path, sid: str) -> dict:
    path = tmp_root / sid / "runflow.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_decide_next_completes_when_no_accounts(tmp_path):
    runs_root = tmp_path / "runs"
    sid = "S-no-accounts"

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


def test_decide_next_completes_when_validation_has_no_findings(tmp_path):
    runs_root = tmp_path / "runs"
    sid = "S-no-findings"

    record_stage(
        sid,
        "merge",
        status="success",
        counts={"count": 2},
        empty_ok=False,
        runs_root=runs_root,
    )
    record_stage(
        sid,
        "validation",
        status="success",
        counts={"findings_count": 0},
        empty_ok=False,
        runs_root=runs_root,
    )

    decision = decide_next(sid, runs_root=runs_root)
    assert decision["next"] == "complete_no_action"
    data = _load_runflow(runs_root, sid)
    assert data["run_state"] == "COMPLETE_NO_ACTION"


def test_decide_next_runs_frontend_then_moves_to_await_input(tmp_path):
    runs_root = tmp_path / "runs"
    sid = "S-with-findings"

    record_stage(
        sid,
        "merge",
        status="success",
        counts={"count": 2},
        empty_ok=False,
        runs_root=runs_root,
    )
    record_stage(
        sid,
        "validation",
        status="success",
        counts={"findings_count": 5},
        empty_ok=False,
        runs_root=runs_root,
    )

    decision = decide_next(sid, runs_root=runs_root)
    assert decision["next"] == "gen_frontend_packs"
    data = _load_runflow(runs_root, sid)
    assert data["run_state"] == "VALIDATING"

    record_stage(
        sid,
        "frontend",
        status="success",
        counts={"packs_count": 2},
        empty_ok=False,
        runs_root=runs_root,
    )

    follow_up = decide_next(sid, runs_root=runs_root)
    assert follow_up["next"] == "await_input"
    data = _load_runflow(runs_root, sid)
    assert data["run_state"] == "AWAITING_CUSTOMER_INPUT"


def test_decide_next_frontend_zero_packs_marks_complete(tmp_path):
    runs_root = tmp_path / "runs"
    sid = "S-frontend-empty"

    record_stage(
        sid,
        "merge",
        status="success",
        counts={"count": 2},
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

    initial = decide_next(sid, runs_root=runs_root)
    assert initial["next"] == "gen_frontend_packs"

    record_stage(
        sid,
        "frontend",
        status="success",
        counts={"packs_count": 0},
        empty_ok=True,
        runs_root=runs_root,
    )

    follow_up = decide_next(sid, runs_root=runs_root)
    assert follow_up == {
        "next": "complete_no_action",
        "reason": "frontend_no_packs",
    }
    data = _load_runflow(runs_root, sid)
    assert data["run_state"] == "COMPLETE_NO_ACTION"


def test_decide_next_stops_on_error(tmp_path):
    runs_root = tmp_path / "runs"
    sid = "S-error"

    record_stage(
        sid,
        "validation",
        status="error",
        counts={"findings_count": 1},
        empty_ok=False,
        runs_root=runs_root,
    )

    decision = decide_next(sid, runs_root=runs_root)
    assert decision["next"] == "stop_error"
    data = _load_runflow(runs_root, sid)
    assert data["run_state"] == "ERROR"
