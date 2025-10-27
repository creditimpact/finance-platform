"""Tests for the lightweight frontend review pack builder."""

from __future__ import annotations

import json
from pathlib import Path

from backend.frontend.review_pack_builder import build_review_packs
from backend.pipeline.runs import RUNS_ROOT_ENV, RunManifest


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def test_build_review_pack_from_manifest(tmp_path: Path, monkeypatch) -> None:
    runs_root = tmp_path / "runs"
    sid = "SID-700"
    monkeypatch.setenv(RUNS_ROOT_ENV, str(runs_root))

    manifest = RunManifest.for_sid(sid)

    account_dir = runs_root / sid / "cases" / "accounts" / "idx-007"
    meta_payload = {"furnisher_name": "Portfolio Recovery Associates, LLC"}
    flat_payload = {
        "account_number_mask": "****1234",
        "account_type": {"value": "Collection"},
        "status_current": "Closed",
    }
    tags_payload = [{"kind": "issue", "type": "collection"}]
    bureaus_payload = {
        "experian": {
            "date_opened": "2018-07",
            "last_payment": "2020-01",
            "dofd": "2019-03",
            "balance": 0,
            "high_balance": 950,
            "credit_limit": 1200,
            "remarks": ["paid collection"],
        },
        "equifax": {
            "date_opened": "2018-07",
            "last_payment": "2020-01",
            "dofd": "2019-03",
            "balance": 0,
            "high_balance": 940,
            "credit_limit": 1200,
            "remarks": [],
        },
        "transunion": {
            "date_opened": "2018-06",
            "last_payment": "2020-02",
            "dofd": "2019-02",
            "balance": 5,
            "high_balance": 955,
            "credit_limit": 1180,
            "remarks": "paid collection",
        },
    }

    _write_json(account_dir / "meta.json", meta_payload)
    _write_json(account_dir / "fields_flat.json", flat_payload)
    _write_json(account_dir / "tags.json", tags_payload)
    _write_json(account_dir / "bureaus.json", bureaus_payload)

    manifest.set_artifact("cases.accounts.idx-007", "dir", account_dir)
    manifest.set_artifact("cases.accounts.idx-007", "meta", account_dir / "meta.json")
    manifest.set_artifact("cases.accounts.idx-007", "flat", account_dir / "fields_flat.json")
    manifest.set_artifact("cases.accounts.idx-007", "tags", account_dir / "tags.json")
    manifest.set_artifact("cases.accounts.idx-007", "bureaus", account_dir / "bureaus.json")

    result = build_review_packs(sid, manifest)

    assert result["status"] == "success"
    packs_dir = Path(result["packs_dir"])
    pack_path = packs_dir / "idx-007.json"
    payload = json.loads(pack_path.read_text(encoding="utf-8"))

    account_section = payload["account"]
    assert account_section["key"] == "idx-007"
    assert account_section["id"] == 7
    assert account_section["furnisher"] == "Portfolio Recovery Associates, LLC"
    assert account_section["account_number"] == "****1234"
    assert account_section["type"] == "Collection"
    assert account_section["status"] == "Closed"
    assert account_section["primary_issue"] == "collection"

    bureau_summary = payload["bureau_summary"]
    assert bureau_summary["opened"] == "2018-07"
    assert bureau_summary["last_payment"] == "2020-01"
    assert bureau_summary["dofd"] == "2019-03"
    assert bureau_summary["balance"] == 5  # picks the max numeric value
    assert bureau_summary["high_balance"] == 955
    assert bureau_summary["limit"] == 1200
    assert "per_bureau" in bureau_summary and len(bureau_summary["per_bureau"]) == 3

    assert payload["attachments_policy"] == {"gov_id_and_poa_default": True}
    assert payload["claims_menu"]

    index_path = Path(result["index"])
    index_payload = json.loads(index_path.read_text(encoding="utf-8"))
    assert index_payload["packs_count"] == 1
    assert index_payload["packs"] == [
        {
            "account_id": "idx-007",
            "path": f"frontend/review/packs/idx-007.json",
            "account_number": "****1234",
            "furnisher": "Portfolio Recovery Associates, LLC",
        }
    ]

