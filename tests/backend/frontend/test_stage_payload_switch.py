import json
import logging
from pathlib import Path

import pytest

from backend.frontend.packs.generator import generate_frontend_packs_for_run


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _build_fields_flat(**fields: dict[str, str]) -> dict[str, object]:
    per_bureau = {}
    for bureau in ("transunion", "experian", "equifax"):
        bureau_values = {}
        for key, value in fields.items():
            if isinstance(value, dict) and bureau in value:
                bureau_values[key] = value[bureau]
        per_bureau[bureau] = bureau_values

    flat: dict[str, object] = {"per_bureau": {}}
    for key, value in fields.items():
        if isinstance(value, dict):
            flat[key] = {"per_bureau": value}
            for bureau, bureau_value in value.items():
                flat["per_bureau"].setdefault(bureau, {})[key] = bureau_value
        else:
            flat[key] = value

    flat.update(per_bureau)
    return flat


def _seed_account(
    runs_root: Path,
    *,
    sid: str,
    account_dir_name: str,
    account_id: str,
    creditor_name: str,
) -> Path:
    account_dir = runs_root / sid / "cases" / "accounts" / account_dir_name
    summary_payload = {
        "account_id": account_id,
        "holder_name": "",  # resolved via fields_flat / labels
        "labels": {
            "creditor": creditor_name,
            "account_type": {"normalized": "Auto Loan"},
            "status": {"normalized": "Closed"},
        },
    }
    flat_payload = _build_fields_flat(
        account_number_display={
            "transunion": "****7007",
            "experian": "****7007",
            "equifax": "****7007",
        },
        account_status={
            "transunion": "Closed",
            "experian": "Closed",
        },
        account_type={
            "transunion": "Auto Loan",
            "experian": "Auto Loan",
        },
        balance_owed={"transunion": "$0"},
        date_opened={"transunion": "2020-01-15"},
        closed_date={"transunion": "2021-04-01"},
        date_reported={"transunion": "2021-05-01"},
    )
    tags_payload = [{"kind": "issue", "type": "wrong_account"}]

    _write_json(account_dir / "summary.json", summary_payload)
    _write_json(account_dir / "fields_flat.json", flat_payload)
    _write_json(account_dir / "tags.json", tags_payload)
    return account_dir


def test_stage_minimal_payload_contains_rich_display(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs_root = tmp_path / "runs"
    sid = "SIDMIN"
    account_id = "idx-007"
    _seed_account(
        runs_root,
        sid=sid,
        account_dir_name="007",
        account_id=account_id,
        creditor_name="Sample Auto Lender",
    )

    monkeypatch.setenv("FRONTEND_STAGE_PAYLOAD", "minimal")

    result = generate_frontend_packs_for_run(sid, runs_root=runs_root)
    assert result["packs_count"] == 1

    pack_path = runs_root / sid / "frontend" / "review" / "packs" / f"{account_id}.json"
    payload = json.loads(pack_path.read_text(encoding="utf-8"))

    assert payload["creditor_name"] == "Sample Auto Lender"
    assert payload["account_type"] == "Auto Loan"
    assert payload["status"] == "Closed"

    display = payload["display"]
    assert display["holder_name"] == "Sample Auto Lender"
    assert display["primary_issue"] == "wrong_account"
    assert display["account_number"]["consensus"] == "****7007"
    assert display["account_type"]["consensus"] == "Auto Loan"
    assert display["status"]["consensus"] == "Closed"
    assert display["balance_owed"]["per_bureau"]["transunion"] == "$0"
    assert display["date_opened"]["transunion"] == "2020-01-15"
    assert display["closed_date"]["transunion"] == "2021-04-01"


def test_stage_full_payload_matches_legacy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs_root = tmp_path / "runs"
    sid = "SIDFULL"
    account_id = "idx-007"
    _seed_account(
        runs_root,
        sid=sid,
        account_dir_name="idx-007",
        account_id=account_id,
        creditor_name="Sample Auto Lender",
    )

    monkeypatch.setenv("FRONTEND_STAGE_PAYLOAD", "full")
    monkeypatch.setenv("FRONTEND_PACKS_DEBUG_MIRROR", "1")

    result = generate_frontend_packs_for_run(sid, runs_root=runs_root)
    assert result["packs_count"] == 1

    packs_dir = runs_root / sid / "frontend" / "review" / "packs"
    pack_path = packs_dir / f"{account_id}.json"
    debug_path = (
        runs_root / sid / "frontend" / "review" / "debug" / f"{account_id}.full.json"
    )

    stage_payload = json.loads(pack_path.read_text(encoding="utf-8"))
    debug_payload = json.loads(debug_path.read_text(encoding="utf-8"))

    assert stage_payload == debug_payload
    assert stage_payload["creditor_name"] == "Sample Auto Lender"
    assert stage_payload["display"]["account_number"]["consensus"] == "****7007"
    assert stage_payload["display"]["account_type"]["consensus"] == "Auto Loan"
    assert stage_payload["display"]["status"]["consensus"] == "Closed"


def test_stage_payload_skip_empty_overwrite_guard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    runs_root = tmp_path / "runs"
    sid = "SIDSAFE"
    account_id = "idx-007"
    account_dir = _seed_account(
        runs_root,
        sid=sid,
        account_dir_name="idx-007",
        account_id=account_id,
        creditor_name="Saved Furnisher",
    )

    stage_pack_path = runs_root / sid / "frontend" / "review" / "packs" / f"{account_id}.json"
    existing_payload = {
        "account_id": account_id,
        "holder_name": "Saved Furnisher",
        "primary_issue": "wrong_account",
        "creditor_name": "Saved Furnisher",
        "account_type": "Auto Loan",
        "status": "Closed",
        "display": {
            "holder_name": "Saved Furnisher",
            "primary_issue": "wrong_account",
            "account_number": {"per_bureau": {"transunion": "****7007"}, "consensus": "****7007"},
        },
    }
    _write_json(stage_pack_path, existing_payload)

    # Replace account inputs with sparse payloads to trigger the guard.
    _write_json(account_dir / "summary.json", {"account_id": account_id, "holder_name": ""})
    _write_json(account_dir / "fields_flat.json", {})
    _write_json(account_dir / "tags.json", [])

    monkeypatch.setenv("FRONTEND_STAGE_PAYLOAD", "minimal")

    caplog.clear()
    with caplog.at_level(logging.INFO, logger="backend.frontend.packs.generator"):
        result = generate_frontend_packs_for_run(sid, runs_root=runs_root)

    assert result["packs_count"] == 1
    assert json.loads(stage_pack_path.read_text(encoding="utf-8")) == existing_payload

    messages = [record.getMessage() for record in caplog.records]
    assert any("PACKGEN_SKIP_EMPTY_OVERWRITE" in message for message in messages)
