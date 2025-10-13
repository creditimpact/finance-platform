import json

from backend.frontend.packs import generator


def test_derive_masked_display_prefers_existing_display() -> None:
    payload = {"display": "XX1234", "last4": "1234"}

    assert generator._derive_masked_display(payload) == "XX1234"


def test_derive_masked_display_masks_from_last4_when_missing_display() -> None:
    payload = {"last4": "9876"}

    assert generator._derive_masked_display(payload) == "****9876"


def test_derive_masked_display_falls_back_to_minimal_mask() -> None:
    assert generator._derive_masked_display(None) == "****"
    assert generator._derive_masked_display({}) == "****"


def test_display_defaults_and_idempotent(tmp_path) -> None:
    runs_root = tmp_path / "runs"
    sid = "SID123"
    account_dir = runs_root / sid / "cases" / "accounts" / "001"
    account_dir.mkdir(parents=True, exist_ok=True)

    summary_path = account_dir / "summary.json"
    summary_path.write_text(json.dumps({"account_id": "001"}), encoding="utf-8")

    bureaus_payload = {
        "transunion": {
            "account_number_display": "",
            "balance_owed": "",
            "date_opened": "",
            "closed_date": "",
        }
    }
    bureaus_path = account_dir / "bureaus.json"
    bureaus_path.write_text(json.dumps(bureaus_payload), encoding="utf-8")

    result = generator.generate_frontend_packs_for_run(sid, runs_root=runs_root)
    assert result["packs_count"] == 1

    pack_path = runs_root / sid / "frontend" / "accounts" / "001" / "pack.json"
    pack_payload = json.loads(pack_path.read_text(encoding="utf-8"))
    display_payload = pack_payload["display"]

    expected_per_bureau = {"transunion": "--", "experian": "--", "equifax": "--"}

    assert display_payload["holder_name"] == ""
    assert display_payload["primary_issue"] == "unknown"
    assert display_payload["account_number"] == {
        "per_bureau": expected_per_bureau,
        "consensus": "--",
    }
    assert display_payload["account_type"] == {
        "per_bureau": expected_per_bureau,
        "consensus": "--",
    }
    assert display_payload["status"] == {
        "per_bureau": expected_per_bureau,
        "consensus": "--",
    }
    assert display_payload["balance_owed"]["per_bureau"] == expected_per_bureau
    assert display_payload["date_opened"] == expected_per_bureau
    assert display_payload["closed_date"] == expected_per_bureau

    index_path = runs_root / sid / "frontend" / "index.json"
    index_payload = json.loads(index_path.read_text(encoding="utf-8"))
    assert index_payload["packs_count"] == 1
    account_entry = index_payload["accounts"][0]
    assert account_entry["holder_name"] == ""
    assert account_entry["primary_issue"] == "unknown"
    assert account_entry["account_number"] == {
        "per_bureau": expected_per_bureau,
        "consensus": "--",
    }
    assert account_entry["account_type"] == {
        "per_bureau": expected_per_bureau,
        "consensus": "--",
    }
    assert account_entry["status"] == {
        "per_bureau": expected_per_bureau,
        "consensus": "--",
    }
    assert account_entry["balance_owed"]["per_bureau"] == expected_per_bureau
    assert account_entry["date_opened"] == expected_per_bureau
    assert account_entry["closed_date"] == expected_per_bureau

    original_pack = pack_path.read_text(encoding="utf-8")
    original_index = index_path.read_text(encoding="utf-8")

    generator.generate_frontend_packs_for_run(sid, runs_root=runs_root)

    assert pack_path.read_text(encoding="utf-8") == original_pack
    assert index_path.read_text(encoding="utf-8") == original_index
