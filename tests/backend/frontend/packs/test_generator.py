from __future__ import annotations

import json
from pathlib import Path

from backend.frontend.packs.generator import generate_frontend_packs_for_run


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_generate_frontend_packs_builds_account_pack(tmp_path):
    runs_root = tmp_path / "runs"
    sid = "S100"
    account_dir = runs_root / sid / "cases" / "accounts" / "1"

    summary_payload = {
        "account_id": "acct-1",
        "labels": {
            "creditor": "Sample Creditor",
            "account_type": {"normalized": "Credit Card"},
            "status": {"normalized": "Closed"},
        },
    }
    bureaus_payload = {
        "transunion": {
            "account_number_display": "****1234",
            "balance_owed": "$100",
            "date_opened": "2023-01-01",
            "closed_date": "2023-02-01",
            "date_reported": "2023-03-01",
            "account_status": "Closed",
            "account_type": "Credit Card",
        },
        "experian": {
            "account_number_display": "XXXX1234",
            "balance_owed": "$100",
            "date_opened": "2023-01-02",
            "closed_date": "--",
            "date_reported": "2023-03-02",
            "account_status": "Closed",
            "account_type": "Credit Card",
        },
    }

    _write_json(account_dir / "summary.json", summary_payload)
    _write_json(account_dir / "bureaus.json", bureaus_payload)

    result = generate_frontend_packs_for_run(sid, runs_root=runs_root)

    pack_path = runs_root / sid / "frontend" / "accounts" / "acct-1" / "pack.json"
    assert pack_path.exists()

    pack_payload = json.loads(pack_path.read_text(encoding="utf-8"))
    assert pack_payload["creditor_name"] == "Sample Creditor"
    assert pack_payload["account_type"] == "Credit Card"
    assert pack_payload["status"] == "Closed"
    assert pack_payload["last4"]["last4"] == "1234"
    assert pack_payload["balance_owed"]["consensus"] == "$100"
    assert set(pack_payload["balance_owed"]["per_bureau"].keys()) == {"transunion", "experian"}
    assert pack_payload["questions"][0]["id"] == "ownership"
    assert len(pack_payload["bureau_badges"]) == 2

    index_path = runs_root / sid / "frontend" / "index.json"
    assert index_path.exists()
    index_payload = json.loads(index_path.read_text(encoding="utf-8"))
    assert index_payload["packs_count"] == 1
    assert index_payload["accounts"][0]["pack_path"] == "frontend/accounts/acct-1/pack.json"
    assert index_payload["questions"][1]["id"] == "recognize"

    responses_dir = runs_root / sid / "frontend" / "responses"
    assert responses_dir.is_dir()
    assert not any(responses_dir.iterdir())

    assert result == {"status": "success", "packs_count": 1, "empty_ok": False}


def test_generate_frontend_packs_handles_missing_accounts(tmp_path):
    runs_root = tmp_path / "runs"
    sid = "S-empty"

    result = generate_frontend_packs_for_run(sid, runs_root=runs_root)

    index_path = runs_root / sid / "frontend" / "index.json"
    assert index_path.exists()
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    assert payload["accounts"] == []
    assert result == {"status": "success", "packs_count": 0, "empty_ok": True}
