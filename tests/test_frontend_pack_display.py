import json
import shutil
from pathlib import Path

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

    stage_dir = runs_root / sid / "frontend" / "review"
    pack_path = stage_dir / "packs" / "001.json"
    pack_payload = json.loads(pack_path.read_text(encoding="utf-8"))
    display_payload = pack_payload["display"]

    expected_per_bureau = {"transunion": "--", "experian": "--", "equifax": "--"}

    assert display_payload["holder_name"] == ""
    assert display_payload["primary_issue"] == "unknown"
    assert display_payload["account_number"] == {"per_bureau": expected_per_bureau}
    assert "consensus" not in display_payload["account_number"]
    assert display_payload["account_type"] == {"per_bureau": expected_per_bureau}
    assert "consensus" not in display_payload["account_type"]
    assert display_payload["status"] == {"per_bureau": expected_per_bureau}
    assert "consensus" not in display_payload["status"]
    assert display_payload["balance_owed"]["per_bureau"] == expected_per_bureau
    assert display_payload["date_opened"] == expected_per_bureau
    assert display_payload["closed_date"] == expected_per_bureau

    index_path = stage_dir / "index.json"
    index_payload = json.loads(index_path.read_text(encoding="utf-8"))
    assert index_payload["packs_count"] == 1
    manifest_entry = index_payload["packs"][0]
    assert manifest_entry["holder_name"] is None
    assert manifest_entry["primary_issue"] == "unknown"
    assert manifest_entry["path"] == "frontend/review/packs/001.json"
    assert manifest_entry["pack_path"] == "frontend/review/packs/001.json"
    manifest_display = manifest_entry["display"]
    assert manifest_display["holder_name"] == ""
    assert manifest_display["primary_issue"] == "unknown"
    assert manifest_display["account_number"] == {
        "per_bureau": expected_per_bureau
    }
    assert manifest_display["account_type"] == {
        "per_bureau": expected_per_bureau
    }
    assert manifest_display["status"] == {"per_bureau": expected_per_bureau}
    assert manifest_display["balance_owed"]["per_bureau"] == expected_per_bureau
    assert manifest_display["date_opened"] == expected_per_bureau
    assert manifest_display["closed_date"] == expected_per_bureau

    original_pack = pack_path.read_text(encoding="utf-8")
    original_index = index_path.read_text(encoding="utf-8")

    generator.generate_frontend_packs_for_run(sid, runs_root=runs_root)

    assert pack_path.read_text(encoding="utf-8") == original_pack
    assert index_path.read_text(encoding="utf-8") == original_index


def test_lean_pack_drops_consensus_defaults() -> None:
    per_bureau = {"transunion": "--", "experian": "--", "equifax": "--"}
    display_payload = {
        "account_number": {"per_bureau": per_bureau, "consensus": "--"},
        "account_type": {"per_bureau": per_bureau, "consensus": "--"},
        "status": {"per_bureau": per_bureau, "consensus": "--"},
        "balance_owed": {"per_bureau": per_bureau},
        "date_opened": per_bureau,
        "closed_date": per_bureau,
    }

    lean_payload = generator.build_lean_pack_doc(
        holder_name="",
        primary_issue="",
        display_payload=display_payload,
        pointers={},
        questions=[],
    )

    display = lean_payload["display"]
    assert display["account_number"] == {"per_bureau": per_bureau}
    assert display["account_type"] == {"per_bureau": per_bureau}
    assert display["status"] == {"per_bureau": per_bureau}


def test_example_account_16_pack_contains_per_bureau(tmp_path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    sample_account_dir = (
        repo_root
        / "runs"
        / "a09686e7-11b9-47a8-a5a0-0fdabc20e220"
        / "cases"
        / "accounts"
        / "16"
    )

    destination_account_dir = (
        tmp_path
        / "runs"
        / "sample-sid"
        / "cases"
        / "accounts"
        / "16"
    )
    destination_account_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(sample_account_dir, destination_account_dir)

    result = generator.generate_frontend_packs_for_run(
        "sample-sid", runs_root=tmp_path / "runs"
    )
    assert result["packs_count"] == 1

    pack_path = (
        tmp_path
        / "runs"
        / "sample-sid"
        / "frontend"
        / "review"
        / "packs"
        / "idx-016.json"
    )
    pack_payload = json.loads(pack_path.read_text(encoding="utf-8"))
    display_payload = pack_payload["display"]

    assert (
        display_payload["account_number"]["per_bureau"]["equifax"]
        == "440066**********"
    )
    assert "consensus" not in display_payload["account_number"]
    assert (
        display_payload["account_type"]["per_bureau"]["transunion"] == "Credit Card"
    )
    assert "consensus" not in display_payload["account_type"]
    assert display_payload["status"]["per_bureau"]["transunion"] == "Closed"
    assert "consensus" not in display_payload["status"]
