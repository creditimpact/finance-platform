from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import pytest

import backend.frontend.packs.generator as generator_module
from backend.domain.claims import CLAIM_FIELD_LINK_MAP
from backend.frontend.packs.generator import generate_frontend_packs_for_run


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _read_stage_pack(base_dir: Path, sid: str, account_id: str) -> tuple[Path, dict]:
    pack_path = base_dir / sid / "frontend" / "review" / "packs" / f"{account_id}.json"
    payload = json.loads(pack_path.read_text(encoding="utf-8"))
    return pack_path, payload


def _build_fields_flat(**fields: object) -> dict:
    per_bureau = {}
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


def test_build_stage_manifest_scans_review_pack_directory(tmp_path: Path) -> None:
    sid = "SID-123"
    run_dir = tmp_path / "runs" / sid
    stage_packs_dir = run_dir / "frontend" / "review" / "packs"
    stage_responses_dir = run_dir / "frontend" / "review" / "responses"
    stage_index_path = run_dir / "frontend" / "review" / "index.json"

    pack_one = {
        "account_id": "idx-001",
        "holder_name": "Alice Example",
        "primary_issue": "wrong_account",
        "questions": [{"id": "ownership"}],
    }
    pack_two = {
        "holder_name": "Bob Example",
        "primary_issue": "identity_theft",
    }

    _write_json(stage_packs_dir / "idx-002.json", pack_two)
    _write_json(stage_packs_dir / "idx-001.json", pack_one)
    stage_responses_dir.mkdir(parents=True, exist_ok=True)

    generator_module._build_stage_manifest(
        sid=sid,
        stage_name="review",
        run_dir=run_dir,
        stage_packs_dir=stage_packs_dir,
        stage_responses_dir=stage_responses_dir,
        stage_index_path=stage_index_path,
    )

    manifest_payload = json.loads(stage_index_path.read_text(encoding="utf-8"))

    assert manifest_payload["sid"] == sid
    assert manifest_payload["stage"] == "review"
    assert manifest_payload["schema_version"] == "1.0"
    assert manifest_payload["responses_dir"] == "frontend/review/responses"
    assert manifest_payload["responses_dir_rel"] == "responses"
    assert manifest_payload["packs_dir"] == "frontend/review/packs"
    assert manifest_payload["packs_dir_rel"] == "packs"
    assert manifest_payload["index_path"] == "frontend/review/index.json"
    assert manifest_payload["index_rel"] == "index.json"
    assert manifest_payload["counts"] == {"packs": 2, "responses": 0}
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", manifest_payload["generated_at"])
    assert manifest_payload["built_at"] == manifest_payload["generated_at"]
    assert manifest_payload["packs_count"] == 2
    assert manifest_payload["questions"] == list(generator_module._QUESTION_SET)
    assert manifest_payload["packs_index"] == [
        {"account": "idx-001", "file": "packs/idx-001.json"},
        {"account": "idx-002", "file": "packs/idx-002.json"},
    ]

    pack_entries = manifest_payload["packs"]
    assert [entry["account_id"] for entry in pack_entries] == ["idx-001", "idx-002"]

    pack_one_path = stage_packs_dir / "idx-001.json"
    pack_two_path = stage_packs_dir / "idx-002.json"

    first_entry, second_entry = pack_entries
    assert first_entry["holder_name"] == "Alice Example"
    assert first_entry["primary_issue"] == "wrong_account"
    assert first_entry["path"] == "frontend/review/packs/idx-001.json"
    assert first_entry["pack_path"] == "frontend/review/packs/idx-001.json"
    assert first_entry["pack_path_rel"] == "packs/idx-001.json"
    assert first_entry["file"] == "frontend/review/packs/idx-001.json"
    assert first_entry["bytes"] == pack_one_path.stat().st_size
    assert first_entry["has_questions"] is True
    assert "display" not in first_entry

    assert second_entry["holder_name"] == "Bob Example"
    assert second_entry["primary_issue"] == "identity_theft"
    assert second_entry["path"] == "frontend/review/packs/idx-002.json"
    assert second_entry["pack_path"] == "frontend/review/packs/idx-002.json"
    assert second_entry["pack_path_rel"] == "packs/idx-002.json"
    assert second_entry["file"] == "frontend/review/packs/idx-002.json"
    assert second_entry["bytes"] == pack_two_path.stat().st_size
    assert "display" not in second_entry


def test_holder_name_from_raw_lines_prefers_spaced_candidate() -> None:
    raw_lines = ["UNRELATED", "JANE SAMPLE", "ACCOUNT # 123"]

    result = generator_module.holder_name_from_raw_lines(raw_lines)

    assert result == "JANE SAMPLE"


def test_holder_name_from_raw_lines_handles_missing_candidates() -> None:
    raw_lines = ["account # 123", "", "12345"]

    result = generator_module.holder_name_from_raw_lines(raw_lines)

    assert result is None


def test_generate_frontend_packs_builds_account_pack(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    sid = "S100"
    account_dir = runs_root / sid / "cases" / "accounts" / "1"

    summary_payload = {
        "account_id": "acct-1",
        "holder_name": "John Doe",
        "labels": {
            "creditor": "Sample Creditor",
            "account_type": {"normalized": "Credit Card"},
            "status": {"normalized": "Closed"},
        },
    }
    flat_payload = _build_fields_flat(
        account_number_display={
            "transunion": "****1234",
            "experian": "XXXX1234",
            "equifax": "",
        },
        balance_owed={
            "transunion": "$100",
            "experian": "$100",
        },
        date_opened={
            "transunion": "2023-01-01",
            "experian": "2023-01-02",
        },
        closed_date={"transunion": "2023-02-01"},
        date_reported={"transunion": "2023-03-01"},
        account_status={
            "transunion": "Closed",
            "experian": "Closed",
        },
        account_type={
            "transunion": "Credit Card",
            "experian": "Credit Card",
        },
    )
    flat_payload["holder_name"] = "John Doe"
    tags_payload = [
        {"kind": "issue", "type": "wrong_account"},
        {"kind": "issue", "type": "late_payment"},
    ]

    _write_json(account_dir / "summary.json", summary_payload)
    _write_json(account_dir / "fields_flat.json", flat_payload)
    _write_json(account_dir / "tags.json", tags_payload)

    result = generate_frontend_packs_for_run(sid, runs_root=runs_root)

    stage_pack_path, stage_pack_payload = _read_stage_pack(runs_root, sid, "acct-1")
    assert list(stage_pack_payload.keys()) == [
        "account_id",
        "holder_name",
        "primary_issue",
        "display",
        "claim_field_links",
        "questions",
    ]
    assert stage_pack_payload["account_id"] == "acct-1"
    assert stage_pack_payload["holder_name"] == "John Doe"
    assert stage_pack_payload["primary_issue"] == "wrong_account"
    assert stage_pack_payload["questions"] == list(generator_module._QUESTION_SET)
    assert stage_pack_payload["claim_field_links"] == CLAIM_FIELD_LINK_MAP

    display_block = stage_pack_payload["display"]
    assert display_block["holder_name"] == "John Doe"
    assert display_block["primary_issue"] == "wrong_account"
    assert display_block["account_number"]["per_bureau"] == {
        "transunion": "****1234",
        "experian": "XXXX1234",
        "equifax": "--",
    }
    assert display_block["account_type"]["per_bureau"] == {
        "transunion": "Credit Card",
        "experian": "Credit Card",
        "equifax": "--",
    }
    assert display_block["status"]["per_bureau"] == {
        "transunion": "Closed",
        "experian": "Closed",
        "equifax": "--",
    }
    assert display_block["balance_owed"]["per_bureau"]["transunion"] == "$100"
    assert display_block["date_opened"]["transunion"] == "2023-01-01"
    assert display_block["closed_date"]["transunion"] == "2023-02-01"

    result_path = runs_root / sid / "frontend" / "review" / "index.json"
    assert result_path.exists()

    manifest_entry = json.loads(result_path.read_text(encoding="utf-8"))["packs"][0]
    assert manifest_entry["display"]["holder_name"] == "John Doe"
    assert manifest_entry["display"]["primary_issue"] == "wrong_account"

    assert result["status"] == "success"
    assert result["packs_count"] == 1
    assert result["empty_ok"] is False


def test_generate_frontend_packs_logs_when_flat_missing(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    runs_root = tmp_path / "runs"
    sid = "S200"
    account_dir = runs_root / sid / "cases" / "accounts" / "1"

    summary_payload = {"account_id": "acct-2", "holder_name": "Case Study"}
    tags_payload = [{"kind": "issue", "type": "identity_theft"}]

    _write_json(account_dir / "summary.json", summary_payload)
    _write_json(account_dir / "tags.json", tags_payload)

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="backend.frontend.packs.generator"):
        result = generate_frontend_packs_for_run(sid, runs_root=runs_root)

    warnings = [record.getMessage() for record in caplog.records]
    assert any("FRONTEND_PACK_MISSING_FLAT" in message for message in warnings)

    assert result["packs_count"] == 1
    stage_pack_path, stage_pack_payload = _read_stage_pack(runs_root, sid, "acct-2")
    assert stage_pack_payload["primary_issue"] == "identity_theft"



def test_generate_frontend_packs_skips_missing_summary(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    runs_root = tmp_path / "runs"
    sid = "S300"
    account_dir = runs_root / sid / "cases" / "accounts" / "1"

    flat_payload = _build_fields_flat(account_number_display={"transunion": "****1111"})
    _write_json(account_dir / "fields_flat.json", flat_payload)
    _write_json(account_dir / "tags.json", [{"kind": "issue", "type": "wrong_account"}])

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="backend.frontend.packs.generator"):
        result = generate_frontend_packs_for_run(sid, runs_root=runs_root)

    warnings = [record.getMessage() for record in caplog.records]
    assert any("FRONTEND_PACK_MISSING_SUMMARY" in message for message in warnings)
    assert result["packs_count"] == 0



def test_generate_frontend_packs_defaults_unknown_issue_when_tags_missing(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    runs_root = tmp_path / "runs"
    sid = "S400"
    account_dir = runs_root / sid / "cases" / "accounts" / "1"

    summary_payload = {"account_id": "acct-4", "holder_name": "No Tags"}
    flat_payload = _build_fields_flat(account_number_display={"experian": "****9999"})

    _write_json(account_dir / "summary.json", summary_payload)
    _write_json(account_dir / "fields_flat.json", flat_payload)

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="backend.frontend.packs.generator"):
        result = generate_frontend_packs_for_run(sid, runs_root=runs_root)

    warnings = [record.getMessage() for record in caplog.records]
    assert any("FRONTEND_PACK_MISSING_TAGS" in message for message in warnings)
    assert result["packs_count"] == 1

    _, pack_payload = _read_stage_pack(runs_root, sid, "acct-4")
    assert pack_payload["primary_issue"] == "unknown"


