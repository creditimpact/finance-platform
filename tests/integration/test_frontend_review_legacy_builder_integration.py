from __future__ import annotations

import json
from pathlib import Path

from backend.frontend.packs.generator import generate_frontend_packs_for_run


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _build_fields_flat(**fields: object) -> dict:
    per_bureau: dict[str, dict[str, object]] = {}
    for bureau in ("transunion", "experian", "equifax"):
        per_bureau[bureau] = {
            key: value[bureau]
            for key, value in fields.items()
            if isinstance(value, dict) and bureau in value
        }

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


def test_generate_frontend_packs_full_pipeline(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    sid = "SID-LEGACY-123"
    account_id = "idx-001"
    account_dir = runs_root / sid / "cases" / "accounts" / "42"

    meta_payload = {"heading_guess": "Example Furnisher"}

    summary_payload = {
        "account_id": account_id,
        "holder_name": "Example Furnisher",
        "labels": {
            "creditor_name": "Example Furnisher",
            "account_type": {"normalized": "Credit Card"},
            "status": {"normalized": "Open"},
        },
    }

    fields_flat_payload = _build_fields_flat(
        account_number_display={
            "transunion": "****1234",
            "experian": "****1234",
            "equifax": "****1234",
        },
        account_type={
            "transunion": "Credit Card",
            "experian": "Credit Card",
            "equifax": "Credit Card",
        },
        account_status={
            "transunion": "Open",
            "experian": "Open",
            "equifax": "Open",
        },
        balance_owed={
            "transunion": "$1,234",
            "experian": "$1,234",
            "equifax": "$1,234",
        },
        date_opened={
            "transunion": "2022-01-01",
            "experian": "2022-01-02",
            "equifax": "2022-01-03",
        },
        closed_date={
            "transunion": "",
            "experian": "",
            "equifax": "",
        },
        date_reported={
            "transunion": "2023-04-05",
            "experian": "2023-04-06",
            "equifax": "2023-04-07",
        },
    )
    fields_flat_payload["holder_name"] = "Example Furnisher"

    tags_payload = [{"kind": "issue", "type": "wrong_account"}]

    bureaus_payload = {
        "transunion": {
            "account_number": "1234",
            "account_type": "Credit Card",
            "account_status": "Open",
            "reported_creditor": "Example Furnisher",
        },
        "experian": {
            "account_number": "1234",
            "account_type": "Credit Card",
            "account_status": "Open",
            "reported_creditor": "Example Furnisher",
        },
        "equifax": {
            "account_number": "1234",
            "account_type": "Credit Card",
            "account_status": "Open",
            "reported_creditor": "Example Furnisher",
        },
    }

    _write_json(account_dir / "meta.json", meta_payload)
    _write_json(account_dir / "summary.json", summary_payload)
    _write_json(account_dir / "fields_flat.json", fields_flat_payload)
    _write_json(account_dir / "tags.json", tags_payload)
    _write_json(account_dir / "bureaus.json", bureaus_payload)

    result = generate_frontend_packs_for_run(sid, runs_root=runs_root)
    assert result["status"] == "success"
    assert result["packs_count"] == 1

    pack_path = runs_root / sid / "frontend" / "review" / "packs" / f"{account_id}.json"
    pack_payload = json.loads(pack_path.read_text(encoding="utf-8"))

    assert pack_payload["holder_name"] == "Example Furnisher"
    display = pack_payload["display"]
    assert display["holder_name"] == "Example Furnisher"
    assert display["account_number"]["per_bureau"] == {
        "transunion": "****1234",
        "experian": "****1234",
        "equifax": "****1234",
    }
    assert display["account_type"]["per_bureau"] == {
        "transunion": "Credit Card",
        "experian": "Credit Card",
        "equifax": "Credit Card",
    }
    assert display["status"]["per_bureau"] == {
        "transunion": "Open",
        "experian": "Open",
        "equifax": "Open",
    }
    assert display["balance_owed"]["per_bureau"] == {
        "transunion": "$1,234",
        "experian": "$1,234",
        "equifax": "$1,234",
    }
    assert display["date_opened"] == {
        "transunion": "2022-01-01",
        "experian": "2022-01-02",
        "equifax": "2022-01-03",
    }

    manifest_path = runs_root / sid / "frontend" / "review" / "index.json"
    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))

    pack_entries = manifest_payload["packs"]
    assert any(entry["account_id"] == account_id for entry in pack_entries)
    packs_index = manifest_payload["packs_index"]
    assert {entry["file"] for entry in packs_index} == {"packs/idx-001.json"}
