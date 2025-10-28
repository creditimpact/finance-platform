from __future__ import annotations

import json
import logging
import re
import os
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
    assert stage_pack_payload["account_id"] == "acct-1"
    assert stage_pack_payload["holder_name"] == "Sample Creditor"
    assert stage_pack_payload["primary_issue"] == "wrong_account"
    assert stage_pack_payload["creditor_name"] == "Sample Creditor"
    assert stage_pack_payload["account_type"] == "Credit Card"
    assert stage_pack_payload["status"] == "Closed"
    assert stage_pack_payload["questions"] == list(generator_module._QUESTION_SET)


def test_bureaus_only_display_builds_from_bureaus(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("FRONTEND_USE_BUREAUS_JSON_ONLY", "1")

    runs_root = tmp_path / "runs"
    sid = "BO-001"
    account_dir = runs_root / sid / "cases" / "accounts" / "1"

    bureaus_payload = {
        "transunion": {
            "account_number_display": "****1111",
            "account_type": "Credit Card",
            "account_status": "Open",
            "balance_owed": "$100",
            "date_opened": "2020-01-01",
            "last_payment": "2023-03-01",
            "date_of_first_delinquency": "2022-06-01",
            "high_balance": "$200",
            "credit_limit": "$500",
            "creditor_remarks": "On time",
        },
        "experian": {
            "account_number_display": "1111****",
            "account_type": "Credit Card",
            "account_status": "Open",
            "balance_owed": "$150",
            "date_opened": "2020-01-02",
            "last_payment": "2023-03-05",
            "date_of_last_activity": "2022-07-15",
            "high_balance": "$220",
            "credit_limit": "$550",
            "creditor_remarks": "Updated",
        },
        "equifax": {
            "account_status": "Closed",
            "balance_owed": "$0",
        },
    }

    _write_json(account_dir / "bureaus.json", bureaus_payload)
    _write_json(account_dir / "meta.json", {"heading_guess": "Example Bank"})
    _write_json(account_dir / "tags.json", [{"kind": "issue", "type": "wrong_account"}])

    result = generate_frontend_packs_for_run(sid, runs_root=runs_root)
    assert result["packs_count"] == 1

    stage_pack_path = (
        runs_root
        / sid
        / "frontend"
        / "review"
        / "packs"
        / "1.json"
    )
    stage_pack_payload = json.loads(stage_pack_path.read_text(encoding="utf-8"))
    display = stage_pack_payload["display"]

    assert display["holder_name"] == "Example Bank"
    assert display["primary_issue"] == "wrong_account"
    assert display["account_number"]["per_bureau"]["transunion"] == "****1111"
    assert display["account_number"]["per_bureau"]["equifax"] == "--"
    assert display["balance"]["per_bureau"]["experian"] == "$150"
    assert display["balance_owed"]["per_bureau"]["transunion"] == "$100"
    assert display["high_balance"]["per_bureau"]["transunion"] == "$200"
    assert display["limit"]["per_bureau"]["experian"] == "$550"
    assert display["remarks"]["per_bureau"]["transunion"] == "On time"
    assert display["opened"]["transunion"] == "2020-01-01"
    assert display["last_payment"]["experian"] == "2023-03-05"
    assert display["dofd"]["experian"] == "2022-07-15"


def test_bureaus_only_display_uses_fallbacks(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("FRONTEND_USE_BUREAUS_JSON_ONLY", "1")

    runs_root = tmp_path / "runs"
    sid = "BO-DOFD"
    account_dir = runs_root / sid / "cases" / "accounts" / "1"

    bureaus_payload = {
        "transunion": {
            "account_number_display": "****2222",
            "account_type": "Collection",
            "payment_status": "Collection/Chargeoff",
            "balance_owed": "$250",
            "date_of_last_activity": "2021-12-15",
        }
    }

    _write_json(account_dir / "bureaus.json", bureaus_payload)
    _write_json(account_dir / "meta.json", {"heading_guess": "Fallback Bank"})
    _write_json(account_dir / "tags.json", [{"kind": "issue", "type": "wrong_account"}])

    result = generate_frontend_packs_for_run(sid, runs_root=runs_root)
    assert result["packs_count"] == 1

    stage_pack_path = (
        runs_root
        / sid
        / "frontend"
        / "review"
        / "packs"
        / "1.json"
    )
    stage_pack_payload = json.loads(stage_pack_path.read_text(encoding="utf-8"))
    display = stage_pack_payload["display"]

    status_block = display["status"]
    assert status_block["per_bureau"]["transunion"] == "Collection"
    assert status_block["consensus"] == "Collection"
    assert display["dofd"]["transunion"] == "2021-12-15"
    assert display["balance_owed"]["per_bureau"].get("experian", "--") == "--"


def test_bureaus_only_meta_prefers_nested_furnisher(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("FRONTEND_USE_BUREAUS_JSON_ONLY", "1")

    runs_root = tmp_path / "runs"
    sid = "BO-NESTED"
    account_dir = runs_root / sid / "cases" / "accounts" / "1"

    bureaus_payload = {
        "transunion": {"creditor_name": "TransUnion Creditor"},
        "experian": {"creditor_name": "Experian Creditor"},
    }
    meta_payload = {"furnisher": {"display_name": "Meta Furnisher"}}
    tags_payload = [{"kind": "issue", "type": "wrong_account"}]

    _write_json(account_dir / "bureaus.json", bureaus_payload)
    _write_json(account_dir / "meta.json", meta_payload)
    _write_json(account_dir / "tags.json", tags_payload)

    result = generate_frontend_packs_for_run(sid, runs_root=runs_root)
    assert result["packs_count"] == 1

    _, stage_pack_payload = _read_stage_pack(runs_root, sid, "1")

    assert stage_pack_payload["holder_name"] == "Meta Furnisher"
    assert stage_pack_payload["creditor_name"] == "Meta Furnisher"
    assert stage_pack_payload["display"]["holder_name"] == "Meta Furnisher"


def test_bureaus_only_holder_falls_back_to_bureaus_when_meta_missing(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("FRONTEND_USE_BUREAUS_JSON_ONLY", "1")

    runs_root = tmp_path / "runs"
    sid = "BO-FALLBACK"
    account_dir = runs_root / sid / "cases" / "accounts" / "1"

    bureaus_payload = {
        "transunion": {"creditor_name": "TransUnion Creditor"},
        "experian": {},
    }
    tags_payload = [{"kind": "issue", "type": "wrong_account"}]

    _write_json(account_dir / "bureaus.json", bureaus_payload)
    _write_json(account_dir / "meta.json", {})
    _write_json(account_dir / "tags.json", tags_payload)

    result = generate_frontend_packs_for_run(sid, runs_root=runs_root)
    assert result["packs_count"] == 1

    _, stage_pack_payload = _read_stage_pack(runs_root, sid, "1")

    assert stage_pack_payload["holder_name"] == "TransUnion Creditor"
    assert stage_pack_payload["creditor_name"] == "TransUnion Creditor"
    assert stage_pack_payload["display"]["holder_name"] == "TransUnion Creditor"


def test_bureaus_only_preserves_issue_priority(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("FRONTEND_USE_BUREAUS_JSON_ONLY", "1")
    monkeypatch.setenv("FRONTEND_STAGE_PAYLOAD", "full")

    runs_root = tmp_path / "runs"
    sid = "BO-ISSUES"
    account_dir = runs_root / sid / "cases" / "accounts" / "1"

    bureaus_payload = {
        "transunion": {
            "account_number_display": "****3333",
            "account_type": "Installment",
            "account_status": "Open",
        }
    }

    tags_payload = [
        {"kind": "note", "type": "skip"},
        {"kind": "issue", "type": "late_payment"},
        {"kind": "issue", "type": "wrong_account"},
        {"kind": "issue", "type": "late_payment"},
    ]

    _write_json(account_dir / "bureaus.json", bureaus_payload)
    _write_json(account_dir / "meta.json", {"heading_guess": "Issue Bank"})
    _write_json(account_dir / "tags.json", tags_payload)

    result = generate_frontend_packs_for_run(sid, runs_root=runs_root)
    assert result["packs_count"] == 1

    _, stage_pack_payload = _read_stage_pack(runs_root, sid, "1")

    assert stage_pack_payload["primary_issue"] == "late_payment"
    assert stage_pack_payload["display"]["primary_issue"] == "late_payment"
    assert stage_pack_payload["issues"] == ["late_payment", "wrong_account"]


def test_generate_frontend_packs_bureaus_only_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs_root = tmp_path / "runs"
    sid = "SBUREAUS"
    account_dir = runs_root / sid / "cases" / "accounts" / "1"

    summary_payload = {"account_id": "acct-1"}
    meta_payload = {"heading_guess": "Bureau Furnisher"}
    bureaus_payload = {
        "transunion": {
            "account_number_display": "****1234",
            "account_type": "Credit Card",
            "account_status": "Open",
            "balance_owed": "$100",
            "date_opened": "2023-01-01",
            "closed_date": "2023-03-01",
        },
        "experian": {
            "account_number_display": "XXXX1234",
            "account_type": "Credit Card",
            "account_status": "Open",
            "balance_owed": "$100",
            "date_opened": "2023-02-01",
            "closed_date": "2023-03-02",
        },
    }
    tags_payload = [{"kind": "issue", "type": "wrong_account"}]

    _write_json(account_dir / "summary.json", summary_payload)
    _write_json(account_dir / "meta.json", meta_payload)
    _write_json(account_dir / "bureaus.json", bureaus_payload)
    _write_json(account_dir / "tags.json", tags_payload)

    monkeypatch.setenv("FRONTEND_USE_BUREAUS_JSON_ONLY", "1")
    monkeypatch.setenv("FRONTEND_STAGE_PAYLOAD", "full")
    monkeypatch.setenv("FRONTEND_PACKS_DEBUG_MIRROR", "0")

    result = generate_frontend_packs_for_run(sid, runs_root=runs_root)

    stage_pack_path, stage_pack_payload = _read_stage_pack(runs_root, sid, "acct-1")
    assert stage_pack_payload["holder_name"] == "Bureau Furnisher"
    assert stage_pack_payload["creditor_name"] == "Bureau Furnisher"
    pointers = stage_pack_payload["pointers"]
    assert "bureaus" in pointers and pointers["bureaus"].endswith("bureaus.json")
    assert "meta" in pointers and pointers["meta"].endswith("meta.json")
    assert "flat" not in pointers

    display = stage_pack_payload["display"]
    assert display["account_number"]["per_bureau"]["transunion"] == "****1234"
    assert display["account_number"]["per_bureau"]["experian"] == "XXXX1234"
    assert display["account_type"]["per_bureau"]["transunion"] == "Credit Card"
    assert display["status"]["per_bureau"]["transunion"] == "Open"
    assert display["balance_owed"]["per_bureau"]["transunion"] == "$100"

    assert result["packs_count"] == 1
    assert stage_pack_payload["claim_field_links"] == CLAIM_FIELD_LINK_MAP
    last4_payload = stage_pack_payload["last4"]
    assert last4_payload["display"] == "****1234"
    assert last4_payload["last4"] == "1234"
    balance_block = stage_pack_payload["balance_owed"]
    assert balance_block["per_bureau"]["transunion"] == "$100"
    dates_block = stage_pack_payload["dates"]
    assert dates_block["date_opened"]["transunion"] == "2023-01-01"
    assert dates_block["closed_date"]["transunion"] == "2023-03-01"
    badges = stage_pack_payload["bureau_badges"]
    assert any(badge["id"] == "transunion" for badge in badges)
    display_block = stage_pack_payload["display"]
    assert display_block["holder_name"] == "Bureau Furnisher"
    assert display_block["primary_issue"] == "wrong_account"
    assert display_block["account_number"]["consensus"] == "****1234"
    assert display_block["account_type"]["consensus"] == "Credit Card"
    assert display_block["status"]["consensus"] == "Open"
    assert display_block["balance_owed"]["per_bureau"]["transunion"] == "$100"
    assert display_block["date_opened"]["transunion"] == "2023-01-01"
    assert display_block["closed_date"]["transunion"] == "2023-03-01"

    result_path = runs_root / sid / "frontend" / "review" / "index.json"
    assert result_path.exists()

    manifest_entry = json.loads(result_path.read_text(encoding="utf-8"))["packs"][0]
    assert manifest_entry["display"]["holder_name"] == "Bureau Furnisher"
    assert manifest_entry["display"]["primary_issue"] == "wrong_account"

    assert result["status"] == "success"
    assert result["packs_count"] == 1
    assert result["empty_ok"] is False


def test_generate_frontend_packs_writes_full_stage_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs_root = tmp_path / "runs"
    sid = "SFULL"
    account_dir = runs_root / sid / "cases" / "accounts" / "1"

    summary_payload = {
        "account_id": "acct-1",
        "holder_name": "Full Case",
        "labels": {
            "creditor": "Sample Creditor",
            "account_type": {"normalized": "Auto Loan"},
            "status": {"normalized": "Closed"},
        },
    }
    flat_payload = _build_fields_flat(
        account_number_display={"transunion": "****0001"},
        balance_owed={"transunion": "$0"},
        account_status={"transunion": "Closed"},
        account_type={"transunion": "Auto Loan"},
    )
    flat_payload["holder_name"] = "Full Case"
    tags_payload = [{"kind": "issue", "type": "wrong_account"}]

    _write_json(account_dir / "summary.json", summary_payload)
    _write_json(account_dir / "fields_flat.json", flat_payload)
    _write_json(account_dir / "tags.json", tags_payload)

    monkeypatch.setenv("FRONTEND_STAGE_PAYLOAD", "full")
    monkeypatch.setenv("FRONTEND_PACKS_DEBUG_MIRROR", "0")
    result = generate_frontend_packs_for_run(sid, runs_root=runs_root)

    stage_pack_path, stage_pack_payload = _read_stage_pack(runs_root, sid, "acct-1")

    assert stage_pack_payload["sid"] == sid
    assert stage_pack_payload["creditor_name"] == "Sample Creditor"
    assert stage_pack_payload["account_type"] == "Auto Loan"
    assert stage_pack_payload["status"] == "Closed"
    pointers = stage_pack_payload["pointers"]
    assert pointers["summary"].endswith("cases/accounts/1/summary.json")
    assert pointers["tags"].endswith("cases/accounts/1/tags.json")
    assert pointers["flat"].endswith("cases/accounts/1/fields_flat.json")
    assert stage_pack_payload["display"]["account_type"]["consensus"] == "Auto Loan"
    assert stage_pack_payload["questions"]
    assert stage_pack_payload["claim_field_links"] == CLAIM_FIELD_LINK_MAP

    debug_dir = runs_root / sid / "frontend" / "review" / "debug"
    debug_files = list(debug_dir.glob("*.full.json")) if debug_dir.exists() else []
    assert not debug_files

    assert result["packs_count"] == 1


def test_generate_frontend_packs_preserves_existing_when_placeholder_payload(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    runs_root = tmp_path / "runs"
    sid = "SPLACE"
    account_dir = runs_root / sid / "cases" / "accounts" / "1"

    summary_payload = {"account_id": "acct-1", "holder_name": ""}
    _write_json(account_dir / "summary.json", summary_payload)
    _write_json(account_dir / "fields_flat.json", {})
    _write_json(account_dir / "tags.json", [])

    stage_pack_path = (
        runs_root / sid / "frontend" / "review" / "packs" / "acct-1.json"
    )
    original_stage_payload = {
        "account_id": "acct-1",
        "holder_name": "Saved Holder",
        "display": {"holder_name": "Saved Holder"},
    }
    _write_json(stage_pack_path, original_stage_payload)

    original_snapshot = json.loads(stage_pack_path.read_text(encoding="utf-8"))

    caplog.clear()
    with caplog.at_level(logging.INFO, logger="backend.frontend.packs.generator"):
        result = generate_frontend_packs_for_run(sid, runs_root=runs_root, force=True)

    updated_payload = json.loads(stage_pack_path.read_text(encoding="utf-8"))
    assert updated_payload["holder_name"] == original_snapshot["holder_name"]
    assert updated_payload["display"]["holder_name"] == original_snapshot["display"]["holder_name"]
    assert result["packs_count"] == 1
    messages = [record.getMessage() for record in caplog.records]
    assert any(
        "PACKGEN_PRESERVED_FIELDS" in message or "PACKGEN_SKIP_EMPTY_OVERWRITE" in message
        for message in messages
    )


def test_generate_frontend_packs_respects_idempotent_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    runs_root = tmp_path / "runs"
    sid = "SLOCK"
    account_dir = runs_root / sid / "cases" / "accounts" / "1"

    summary_payload = {"account_id": "acct-1", "holder_name": "Locked Holder"}
    flat_payload = _build_fields_flat(account_number_display={"transunion": "****9999"})

    _write_json(account_dir / "summary.json", summary_payload)
    _write_json(account_dir / "fields_flat.json", flat_payload)
    _write_json(account_dir / "tags.json", [])

    stage_pack_path = (
        runs_root / sid / "frontend" / "review" / "packs" / "acct-1.json"
    )
    original_payload = {
        "account_id": "acct-1",
        "holder_name": "Locked Holder",
        "display": {"holder_name": "Locked Holder"},
    }
    _write_json(stage_pack_path, original_payload)

    lock_rel = Path("frontend/.locks/idempotent.lock")
    lock_path = runs_root / sid / lock_rel
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text("lock", encoding="utf-8")

    # Ensure the pack is newer than the lock so the writer skips rewriting.
    os.utime(lock_path, (1_000_000, 1_000_000))
    os.utime(stage_pack_path, (1_000_100, 1_000_100))

    monkeypatch.setenv("FRONTEND_IDEMPOTENT_LOCK_REL", lock_rel.as_posix())

    caplog.clear()
    with caplog.at_level(logging.INFO, logger="backend.frontend.packs.generator"):
        result = generate_frontend_packs_for_run(sid, runs_root=runs_root)

    payload_after = json.loads(stage_pack_path.read_text(encoding="utf-8"))
    assert payload_after == original_payload
    assert result["packs_count"] == 1

    messages = [record.getMessage() for record in caplog.records]
    assert any("PACKGEN_SKIP_LOCKED" in message for message in messages)
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


def test_generate_frontend_packs_creates_empty_index_when_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs_root = tmp_path / "runs"
    sid = "S500"

    monkeypatch.setenv("ENABLE_FRONTEND_PACKS", "0")
    monkeypatch.setenv("FRONTEND_REVIEW_CREATE_EMPTY_INDEX", "1")

    try:
        result = generate_frontend_packs_for_run(sid, runs_root=runs_root)
    finally:
        monkeypatch.delenv("ENABLE_FRONTEND_PACKS", raising=False)
        monkeypatch.delenv("FRONTEND_REVIEW_CREATE_EMPTY_INDEX", raising=False)

    index_path = runs_root / sid / "frontend" / "review" / "index.json"
    assert index_path.exists()

    manifest = json.loads(index_path.read_text(encoding="utf-8"))
    assert manifest["packs_count"] == 0
    assert result["packs_count"] == 0
    assert result["last_built_at"] == manifest["generated_at"]

