from __future__ import annotations

import json
from pathlib import Path

from backend.frontend.packs.generator import generate_frontend_packs_for_run
from backend.runflow.decider import decide_next, record_stage


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_runflow_completes_when_validation_has_no_findings(tmp_path):
    runs_root = tmp_path / "runs"
    sid = "SID-no-findings"

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
    assert decision == {"next": "complete_no_action", "reason": "validation_no_findings"}

    result = generate_frontend_packs_for_run(sid, runs_root=runs_root)
    assert result["packs_count"] == 0

    index_path = runs_root / sid / "frontend" / "index.json"
    assert index_path.exists()
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    assert payload["accounts"] == []


def test_runflow_generates_frontend_and_moves_to_await(tmp_path):
    runs_root = tmp_path / "runs"
    sid = "SID-with-findings"

    # Prepare two accounts for the generator.
    for account_id in ("acct-1", "acct-2"):
        account_dir = runs_root / sid / "cases" / "accounts" / account_id
        _write_json(
            account_dir / "summary.json",
            {
                "account_id": account_id,
                "labels": {"creditor": f"Creditor {account_id}"},
            },
        )
        _write_json(
            account_dir / "bureaus.json",
            {
                "transunion": {
                    "account_number_display": f"****{account_id[-1]}001",
                    "account_type": "Credit Card",
                    "account_status": "Open",
                }
            },
        )

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
        counts={"findings_count": 2},
        empty_ok=False,
        runs_root=runs_root,
    )

    decision = decide_next(sid, runs_root=runs_root)
    assert decision == {"next": "gen_frontend_packs", "reason": "validation_has_findings"}

    result = generate_frontend_packs_for_run(sid, runs_root=runs_root)
    assert result["packs_count"] == 2

    record_stage(
        sid,
        "frontend",
        status="success",
        counts={"packs_count": 2},
        empty_ok=False,
        runs_root=runs_root,
    )

    follow_up = decide_next(sid, runs_root=runs_root)
    assert follow_up == {"next": "await_input", "reason": "frontend_completed"}

    index_path = runs_root / sid / "frontend" / "index.json"
    assert index_path.exists()
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    assert len(payload["accounts"]) == 2
