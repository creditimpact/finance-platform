from __future__ import annotations

import importlib
import json
from pathlib import Path
import re

import backend.frontend.packs.generator as generator_module
from backend.frontend.packs.generator import generate_frontend_packs_for_run
from backend.frontend.packs.responses import append_frontend_response


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _read_stage_pack(base_dir: Path, sid: str, account_id: str) -> tuple[Path, dict]:
    pack_path = base_dir / sid / "frontend" / "review" / "packs" / f"{account_id}.json"
    payload = json.loads(pack_path.read_text(encoding="utf-8"))
    return pack_path, payload


def test_build_stage_manifest_scans_review_pack_directory(tmp_path):
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
    assert manifest_payload["counts"] == {"packs": 2, "responses": 0}
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", manifest_payload["generated_at"])

    pack_entries = manifest_payload["packs"]
    assert [entry["account_id"] for entry in pack_entries] == ["idx-001", "idx-002"]

    pack_one_path = stage_packs_dir / "idx-001.json"
    pack_two_path = stage_packs_dir / "idx-002.json"

    first_entry, second_entry = pack_entries
    assert first_entry["holder_name"] == "Alice Example"
    assert first_entry["primary_issue"] == "wrong_account"
    assert first_entry["path"] == "frontend/review/packs/idx-001.json"
    assert first_entry["bytes"] == pack_one_path.stat().st_size
    assert first_entry["has_questions"] is True

    assert second_entry["holder_name"] == "Bob Example"
    assert second_entry["primary_issue"] == "identity_theft"
    assert second_entry["path"] == "frontend/review/packs/idx-002.json"
    assert second_entry["bytes"] == pack_two_path.stat().st_size
    assert second_entry["has_questions"] is True


def test_holder_name_from_raw_lines_prefers_spaced_candidate() -> None:
    raw_lines = ["UNRELATED", "JANE SAMPLE", "ACCOUNT # 123"]

    result = generator_module.holder_name_from_raw_lines(raw_lines)

    assert result == "JANE SAMPLE"


def test_holder_name_from_raw_lines_handles_missing_candidates() -> None:
    raw_lines = ["account # 123", "", "12345"]

    result = generator_module.holder_name_from_raw_lines(raw_lines)

    assert result is None


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
    meta_payload = {"heading_guess": "John Doe"}
    raw_lines_payload = [
        {"text": "JOHN DOE"},
        {"text": "Account # 1234"},
    ]
    tags_payload = [
        {"kind": "issue", "type": "wrong_account"},
        {"kind": "note", "type": "internal"},
        {"kind": "issue", "type": "late_payment"},
    ]

    _write_json(account_dir / "summary.json", summary_payload)
    _write_json(account_dir / "bureaus.json", bureaus_payload)
    _write_json(account_dir / "meta.json", meta_payload)
    _write_json(account_dir / "raw_lines.json", raw_lines_payload)
    _write_json(account_dir / "tags.json", tags_payload)

    result = generate_frontend_packs_for_run(sid, runs_root=runs_root)

    stage_pack_path, stage_pack_payload = _read_stage_pack(runs_root, sid, "acct-1")
    assert list(stage_pack_payload.keys()) == [
        "account_id",
        "holder_name",
        "primary_issue",
        "display",
    ]
    assert stage_pack_payload["account_id"] == "acct-1"
    assert stage_pack_payload["holder_name"] == "John Doe"
    assert stage_pack_payload["primary_issue"] == "wrong_account"
    assert "questions" not in stage_pack_payload
    assert "pointers" not in stage_pack_payload

    display_block = stage_pack_payload["display"]
    assert list(display_block.keys()) == [
        "display_version",
        "holder_name",
        "primary_issue",
        "account_number",
        "account_type",
        "status",
        "balance_owed",
        "date_opened",
        "closed_date",
    ]
    assert display_block["display_version"] == generator_module._DISPLAY_SCHEMA_VERSION
    assert display_block["holder_name"] == "John Doe"
    assert display_block["primary_issue"] == "wrong_account"
    assert display_block["account_number"] == {
        "per_bureau": {
            "transunion": "****1234",
            "experian": "XXXX1234",
            "equifax": "--",
        }
    }
    assert display_block["account_type"] == {
        "per_bureau": {
            "transunion": "Credit Card",
            "experian": "Credit Card",
            "equifax": "--",
        }
    }
    assert display_block["status"] == {
        "per_bureau": {
            "transunion": "Closed",
            "experian": "Closed",
            "equifax": "--",
        }
    }
    assert display_block["balance_owed"] == {
        "per_bureau": {
            "transunion": "$100",
            "experian": "$100",
            "equifax": "--",
        }
    }
    assert display_block["date_opened"] == {
        "transunion": "2023-01-01",
        "experian": "2023-01-02",
        "equifax": "--",
    }
    assert display_block["closed_date"] == {
        "transunion": "2023-02-01",
        "experian": "--",
        "equifax": "--",
    }

    index_path = runs_root / sid / "frontend" / "index.json"
    assert index_path.exists()
    index_payload = json.loads(index_path.read_text(encoding="utf-8"))
    assert index_payload["schema_version"] == generator_module._FRONTEND_INDEX_SCHEMA_VERSION
    assert index_payload["packs_count"] == 1
    index_entry = index_payload["accounts"][0]
    assert list(index_entry.keys()) == [
        "account_id",
        "holder_name",
        "primary_issue",
        "account_number",
        "account_type",
        "status",
        "balance_owed",
        "date_opened",
        "closed_date",
        "pack_path",
    ]
    assert index_entry["pack_path"] == "frontend/review/packs/acct-1.json"
    assert index_entry["holder_name"] == "John Doe"
    assert index_entry["primary_issue"] == "wrong_account"
    assert (
        index_entry["account_number"]["per_bureau"]
        == display_block["account_number"]["per_bureau"]
    )
    assert (
        index_entry["account_type"]["per_bureau"]
        == display_block["account_type"]["per_bureau"]
    )
    assert (
        index_entry["status"]["per_bureau"]
        == display_block["status"]["per_bureau"]
    )
    assert index_entry["balance_owed"] == display_block["balance_owed"]
    assert index_entry["date_opened"] == display_block["date_opened"]
    assert index_entry["closed_date"] == display_block["closed_date"]
    assert index_payload["questions"][1]["id"] == "recognize"

    stage_index_path = runs_root / sid / "frontend" / "review" / "index.json"
    assert stage_index_path.exists()
    stage_index_payload = json.loads(stage_index_path.read_text(encoding="utf-8"))
    assert stage_index_payload["stage"] == "review"
    assert stage_index_payload["counts"]["packs"] == 1
    assert stage_index_payload["counts"]["responses"] == 0
    manifest_entry = stage_index_payload["packs"][0]
    assert manifest_entry["account_id"] == "acct-1"
    assert manifest_entry["holder_name"] == "John Doe"
    assert manifest_entry["primary_issue"] == "wrong_account"
    assert manifest_entry["path"] == "frontend/review/packs/acct-1.json"

    responses_dir = runs_root / sid / "frontend" / "review" / "responses"
    assert responses_dir.is_dir()
    assert not any(responses_dir.iterdir())

    assert result["status"] == "success"
    assert result["packs_count"] == 1
    assert result["empty_ok"] is False
    assert result["built"] is True
    assert result["packs_dir"] == str(
        (runs_root / sid / "frontend" / "review" / "packs").absolute()
    )
    assert isinstance(result["last_built_at"], str)


def test_frontend_review_stage_minimal_smoke(tmp_path):
    runs_root = tmp_path / "runs"
    sid = "SID-frontend-review-smoke"

    for index in range(2):
        account_id = f"acct-{index + 1}"
        account_dir = runs_root / sid / "cases" / "accounts" / str(index)
        summary_payload = {
            "account_id": account_id,
            "labels": {
                "creditor": f"Creditor {index + 1}",
                "account_type": {"normalized": "Credit Card"},
                "status": {"normalized": "Open"},
            },
        }
        bureaus_payload = {
            "transunion": {
                "account_number_display": f"****{index + 1}234",
                "balance_owed": "$123",
                "date_opened": "2024-01-01",
                "account_status": "Open",
                "account_type": "Credit Card",
            }
        }
        _write_json(account_dir / "summary.json", summary_payload)
        _write_json(account_dir / "bureaus.json", bureaus_payload)

    result = generate_frontend_packs_for_run(sid, runs_root=runs_root)
    assert result["packs_count"] == 2

    run_dir = runs_root / sid
    stage_dir = run_dir / "frontend" / "review"
    stage_packs_dir = stage_dir / "packs"
    stage_responses_dir = stage_dir / "responses"

    assert stage_dir.is_dir()
    assert stage_packs_dir.is_dir()
    assert stage_responses_dir.is_dir()

    pack_files = sorted(stage_packs_dir.glob("*.json"))
    assert len(pack_files) == 2

    manifest_path = stage_dir / "index.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["counts"]["packs"] == 2

    manifest_entries = {entry["path"]: entry for entry in manifest["packs"]}
    assert set(manifest_entries) == {
        f"frontend/review/packs/{path.name}" for path in pack_files
    }

    for pack_path in pack_files:
        payload = json.loads(pack_path.read_text(encoding="utf-8"))
        assert set(payload) == {"account_id", "holder_name", "primary_issue", "display"}

        rel_path = f"frontend/review/packs/{pack_path.name}"
        entry = manifest_entries[rel_path]
        assert entry["bytes"] == pack_path.stat().st_size
        assert entry["path"] == rel_path

    append_frontend_response(run_dir, "acct-1", {"answer": "yes"})
    append_frontend_response(run_dir, "acct-1", {"answer": "still yes"})

    response_path = stage_responses_dir / "acct-1.jsonl"
    assert response_path.exists()
    response_lines = response_path.read_text(encoding="utf-8").splitlines()
    assert response_lines == ["{\"answer\": \"yes\"}", "{\"answer\": \"still yes\"}"]


def test_frontend_runflow_steps_are_condensed(tmp_path, monkeypatch):
    runs_root = tmp_path / "runs"
    sid = "S200"

    monkeypatch.setenv("RUNFLOW_VERBOSE", "1")
    monkeypatch.setenv("RUNS_ROOT", str(runs_root))

    import backend.core.runflow as runflow_module

    runflow_module = importlib.reload(runflow_module)
    generator = importlib.reload(generator_module)

    try:
        account_dir = runs_root / sid / "cases" / "accounts" / "1"

        summary_payload = {
            "account_id": "acct-1",
            "labels": {
                "account_type": {"normalized": "Conventional real estate mortgage"},
                "status": {"normalized": "Open"},
            },
        }
        bureaus_payload = {
            "transunion": {
                "account_number_display": "277003*******",
                "balance_owed": "$1,155,606",
                "date_opened": "22.2.2022",
                "closed_date": "--",
                "account_status": "Open",
                "account_type": "Conventional real estate mortgage",
            },
            "experian": {
                "account_number_display": "277003*******",
                "balance_owed": "$1,155,606",
                "date_opened": "1.2.2022",
                "closed_date": "--",
                "account_status": "Open",
                "account_type": "Conventional real estate mortgage",
            },
            "equifax": {
                "account_number_display": "277003*******",
                "balance_owed": "$1,155,606",
                "date_opened": "1.2.2022",
                "closed_date": "--",
                "account_status": "Open",
                "account_type": "Conventional real estate mortgage",
            },
        }
        meta_payload = {"heading_guess": "SPS"}
        raw_lines_payload = [{"text": "SPS"}]
        tags_payload = [{"kind": "issue", "type": "delinquency"}]

        _write_json(account_dir / "summary.json", summary_payload)
        _write_json(account_dir / "bureaus.json", bureaus_payload)
        _write_json(account_dir / "meta.json", meta_payload)
        _write_json(account_dir / "raw_lines.json", raw_lines_payload)
        _write_json(account_dir / "tags.json", tags_payload)

        result = generator.generate_frontend_packs_for_run(sid, runs_root=runs_root)

        stage_pack_path, stage_pack_payload = _read_stage_pack(runs_root, sid, "acct-1")
        assert stage_pack_path.exists()
        display_block = stage_pack_payload["display"]
        assert list(display_block.keys()) == [
            "display_version",
            "holder_name",
            "primary_issue",
            "account_number",
            "account_type",
            "status",
            "balance_owed",
            "date_opened",
            "closed_date",
        ]
        assert display_block["display_version"] == generator_module._DISPLAY_SCHEMA_VERSION

        index_path = runs_root / sid / "frontend" / "index.json"
        assert index_path.exists()
        index_payload = json.loads(index_path.read_text(encoding="utf-8"))
        assert index_payload["packs_count"] == 1
        index_entry = index_payload["accounts"][0]
        assert index_entry["pack_path"] == "frontend/review/packs/acct-1.json"

        steps_path = runs_root / sid / "runflow_steps.json"
        assert steps_path.exists()
        steps_payload = json.loads(steps_path.read_text(encoding="utf-8"))
        frontend_steps = steps_payload["stages"]["frontend"]["substages"]["default"]["steps"]
        step_names = [step["name"] for step in frontend_steps]
        assert step_names == [
            "frontend_review_start",
            "frontend_review_pack_created",
            "frontend_review_finish",
        ]

        start_step = next(
            step for step in frontend_steps if step["name"] == "frontend_review_start"
        )
        assert start_step.get("metrics") == {"accounts": 1}

        pack_created_step = next(
            step for step in frontend_steps if step["name"] == "frontend_review_pack_created"
        )
        pack_out = pack_created_step.get("out")
        assert pack_out is not None
        assert pack_out["account_id"] == "acct-1"
        assert pack_out["path"] == "frontend/review/packs/acct-1.json"
        assert isinstance(pack_out["bytes"], int)
        assert pack_out["bytes"] > 0

        assert result["status"] == "success"
        assert result["packs_count"] == 1
    finally:
        monkeypatch.delenv("RUNFLOW_VERBOSE", raising=False)
        monkeypatch.delenv("RUNS_ROOT", raising=False)
        importlib.reload(runflow_module)
        importlib.reload(generator_module)


def test_generate_frontend_packs_falls_back_to_raw_holder_name(tmp_path):
    runs_root = tmp_path / "runs"
    sid = "S101"
    account_dir = runs_root / sid / "cases" / "accounts" / "1"

    summary_payload = {"account_id": "acct-raw"}
    bureaus_payload = {
        "transunion": {
            "account_number_display": "****9876",
            "balance_owed": "$200",
            "date_opened": "2023-04-01",
            "closed_date": "--",
            "account_status": "Open",
            "account_type": "Mortgage",
        }
    }
    raw_lines_payload = [
        {"text": "UNRELATED"},
        {"text": "JANE SAMPLE"},
    ]
    tags_payload = [{"kind": "issue", "type": "late_payment"}]

    _write_json(account_dir / "summary.json", summary_payload)
    _write_json(account_dir / "bureaus.json", bureaus_payload)
    _write_json(account_dir / "meta.json", {})
    _write_json(account_dir / "raw_lines.json", raw_lines_payload)
    _write_json(account_dir / "tags.json", tags_payload)

    generate_frontend_packs_for_run(sid, runs_root=runs_root)

    _, pack_payload = _read_stage_pack(runs_root, sid, "acct-raw")

    assert pack_payload["holder_name"] == "JANE SAMPLE"
    assert pack_payload["display"]["holder_name"] == "JANE SAMPLE"


def test_generate_frontend_packs_defaults_unknown_issue(tmp_path):
    runs_root = tmp_path / "runs"
    sid = "S102"
    account_dir = runs_root / sid / "cases" / "accounts" / "1"

    summary_payload = {"account_id": "acct-issue"}
    bureaus_payload = {
        "transunion": {
            "account_number_display": "****1357",
            "balance_owed": "$500",
            "date_opened": "2022-12-01",
            "closed_date": "--",
            "account_status": "Open",
            "account_type": "Loan",
        }
    }
    meta_payload = {"heading_guess": "ACME BANK"}
    raw_lines_payload = [{"text": "ACME BANK"}]

    _write_json(account_dir / "summary.json", summary_payload)
    _write_json(account_dir / "bureaus.json", bureaus_payload)
    _write_json(account_dir / "meta.json", meta_payload)
    _write_json(account_dir / "raw_lines.json", raw_lines_payload)
    _write_json(account_dir / "tags.json", [])

    generate_frontend_packs_for_run(sid, runs_root=runs_root)

    _, pack_payload = _read_stage_pack(runs_root, sid, "acct-issue")

    assert pack_payload["primary_issue"] == "unknown"
    assert pack_payload["display"]["primary_issue"] == "unknown"


def _write_minimal_account(
    account_dir: Path,
    *,
    account_id: str,
    meta_payload: dict | None = None,
    raw_lines_payload: list | None = None,
    tags_payload: list | None = None,
) -> None:
    summary_payload = {"account_id": account_id}
    bureaus_payload = {
        "transunion": {
            "account_number_display": "****4321",
            "balance_owed": "$123",
            "date_opened": "2023-01-01",
            "date_reported": "2023-02-01",
            "account_status": "Open",
            "account_type": "Loan",
        }
    }

    _write_json(account_dir / "summary.json", summary_payload)
    _write_json(account_dir / "bureaus.json", bureaus_payload)
    if meta_payload is not None:
        _write_json(account_dir / "meta.json", meta_payload)
    if raw_lines_payload is not None:
        _write_json(account_dir / "raw_lines.json", raw_lines_payload)
    if tags_payload is not None:
        _write_json(account_dir / "tags.json", tags_payload)


def test_generate_frontend_packs_meta_heading_and_primary_issue(tmp_path):
    runs_root = tmp_path / "runs"
    sid = "S-meta"
    account_dir = runs_root / sid / "cases" / "accounts" / "1"

    _write_minimal_account(
        account_dir,
        account_id="acct-meta",
        meta_payload={"heading_guess": "SPS"},
        raw_lines_payload=[{"text": "SPS"}],
        tags_payload=[{"kind": "issue", "type": "delinquency"}],
    )

    result = generate_frontend_packs_for_run(sid, runs_root=runs_root)

    pack_path, payload = _read_stage_pack(runs_root, sid, "acct-meta")

    assert list(payload.keys()) == [
        "account_id",
        "holder_name",
        "primary_issue",
        "display",
    ]
    assert payload["account_id"] == "acct-meta"
    assert payload["holder_name"] == "SPS"
    assert payload["primary_issue"] == "delinquency"

    debug_dir = runs_root / sid / "frontend" / "review" / "debug"
    if debug_dir.exists():
        assert not any(debug_dir.iterdir())

    assert result["packs_count"] == 1


def test_generate_frontend_packs_holder_name_from_raw_when_meta_missing(tmp_path):
    runs_root = tmp_path / "runs"
    sid = "S-raw"
    account_dir = runs_root / sid / "cases" / "accounts" / "1"

    _write_minimal_account(
        account_dir,
        account_id="acct-raw",
        raw_lines_payload=[{"text": "ACME BANK"}, {"text": "ACCOUNT #123"}],
        tags_payload=[{"kind": "issue", "type": "ownership"}],
    )

    result = generate_frontend_packs_for_run(sid, runs_root=runs_root)

    _, payload = _read_stage_pack(runs_root, sid, "acct-raw")

    assert list(payload.keys()) == [
        "account_id",
        "holder_name",
        "primary_issue",
        "display",
    ]
    assert payload["holder_name"] == "ACME BANK"
    assert payload["primary_issue"] == "ownership"

    assert result["packs_count"] == 1


def test_generate_frontend_packs_backfills_missing_pointers(tmp_path, monkeypatch):
    runs_root = tmp_path / "runs"
    sid = "S-backfill"
    account_dir = runs_root / sid / "cases" / "accounts" / "1"

    monkeypatch.setenv("FRONTEND_PACKS_LEAN", "0")

    _write_minimal_account(
        account_dir,
        account_id="acct-legacy",
        tags_payload=[{"kind": "issue", "type": "collection"}],
    )

    legacy_pack_dir = runs_root / sid / "frontend" / "accounts" / "acct-legacy"
    legacy_pack_payload = {
        "sid": sid,
        "account_id": "acct-legacy",
        "creditor_name": "Legacy Creditor",
        "account_type": "Loan",
        "status": "Open",
        "last4": {"display": "****4321", "last4": "4321"},
        "balance_owed": {"consensus": "$123", "per_bureau": {"transunion": "$123"}},
        "dates": {},
        "bureau_badges": [],
        "holder_name": "Legacy Creditor",
        "primary_issue": "collection",
        "questions": generator_module._QUESTION_SET,
    }
    _write_json(legacy_pack_dir / "pack.json", legacy_pack_payload)

    legacy_index_payload = {
        "sid": sid,
        "accounts": [
            {
                "account_id": "acct-legacy",
                "pack_path": "frontend/accounts/acct-legacy/pack.json",
                "creditor_name": "Legacy Creditor",
                "account_type": "Loan",
                "status": "Open",
                "holder_name": "Legacy Creditor",
                "primary_issue": "collection",
                "balance_owed": "$123",
                "bureau_badges": [],
            }
        ],
        "packs_count": 1,
        "questions": generator_module._QUESTION_SET,
        "generated_at": "2023-01-01T00:00:00Z",
    }
    _write_json(runs_root / sid / "frontend" / "index.json", legacy_index_payload)

    result = generate_frontend_packs_for_run(sid, runs_root=runs_root)

    stage_pack_path, stage_payload = _read_stage_pack(runs_root, sid, "acct-legacy")
    assert set(stage_payload.keys()) == {
        "account_id",
        "holder_name",
        "primary_issue",
        "display",
    }
    assert stage_payload["account_id"] == "acct-legacy"
    assert "pointers" not in stage_payload
    assert "questions" not in stage_payload

    index_payload = json.loads(
        (runs_root / sid / "frontend" / "index.json").read_text(encoding="utf-8")
    )
    assert index_payload["accounts"][0]["pack_path"] == "frontend/review/packs/acct-legacy.json"
    assert result["packs_count"] == 1


def test_generate_frontend_packs_debug_mirror_toggle(tmp_path, monkeypatch):
    runs_root = tmp_path / "runs"
    sid = "S-debug"
    account_dir = runs_root / sid / "cases" / "accounts" / "1"

    _write_minimal_account(
        account_dir,
        account_id="acct-debug",
        meta_payload={"heading_guess": "Debug Holder"},
        tags_payload=[{"kind": "issue", "type": "collection"}],
    )

    monkeypatch.setenv("FRONTEND_PACKS_DEBUG_MIRROR", "1")

    generate_frontend_packs_for_run(sid, runs_root=runs_root)

    _, pack_payload = _read_stage_pack(runs_root, sid, "acct-debug")
    debug_dir = runs_root / sid / "frontend" / "review" / "debug"
    mirror_path = debug_dir / "acct-debug.full.json"

    assert mirror_path.exists()
    mirror_payload = json.loads(mirror_path.read_text(encoding="utf-8"))

    assert list(pack_payload.keys()) == [
        "account_id",
        "holder_name",
        "primary_issue",
        "display",
    ]
    assert mirror_payload["sid"] == sid
    assert mirror_payload["account_id"] == "acct-debug"
    assert mirror_payload["holder_name"] == "Debug Holder"
    assert mirror_payload["questions"] == list(generator_module._QUESTION_SET)

    monkeypatch.setenv("FRONTEND_PACKS_DEBUG_MIRROR", "0")

    generate_frontend_packs_for_run(sid, runs_root=runs_root)

    assert not mirror_path.exists()

def test_generate_frontend_packs_multiple_issues_first_primary(tmp_path):
    runs_root = tmp_path / "runs"
    sid = "S-issues"
    account_dir = runs_root / sid / "cases" / "accounts" / "1"

    _write_minimal_account(
        account_dir,
        account_id="acct-issues",
        tags_payload=[
            {"kind": "issue", "type": "collection"},
            {"kind": "issue", "type": "late_history"},
        ],
    )

    result = generate_frontend_packs_for_run(sid, runs_root=runs_root)

    _, payload = _read_stage_pack(runs_root, sid, "acct-issues")

    assert list(payload.keys()) == [
        "account_id",
        "holder_name",
        "primary_issue",
        "display",
    ]
    assert payload["primary_issue"] == "collection"

    assert result["packs_count"] == 1

def test_generate_frontend_packs_holder_name_fallback(tmp_path):
    runs_root = tmp_path / "runs"
    sid = "S101"
    account_dir = runs_root / sid / "cases" / "accounts" / "2"

    summary_payload = {"account_id": "acct-2"}
    bureaus_payload = {
        "equifax": {
            "account_number_display": "****5678",
            "balance_owed": "$75",
            "account_status": "Open",
            "account_type": "Loan",
        }
    }
    raw_lines_payload = [
        "UNRELATED",
        {"text": "JANE SAMPLE"},
    ]
    tags_payload = [{"kind": "issue", "type": "identity_theft"}]

    _write_json(account_dir / "summary.json", summary_payload)
    _write_json(account_dir / "bureaus.json", bureaus_payload)
    _write_json(account_dir / "raw_lines.json", raw_lines_payload)
    _write_json(account_dir / "tags.json", tags_payload)

    result = generate_frontend_packs_for_run(sid, runs_root=runs_root)

    _, pack_payload = _read_stage_pack(runs_root, sid, "acct-2")

    assert list(pack_payload.keys()) == [
        "account_id",
        "holder_name",
        "primary_issue",
        "display",
    ]
    assert pack_payload["holder_name"] == "JANE SAMPLE"
    assert pack_payload["primary_issue"] == "identity_theft"

    index_path = runs_root / sid / "frontend" / "index.json"
    index_payload = json.loads(index_path.read_text(encoding="utf-8"))
    assert index_payload["schema_version"] == generator_module._FRONTEND_INDEX_SCHEMA_VERSION
    index_entry = index_payload["accounts"][0]

    assert index_entry["holder_name"] == "JANE SAMPLE"
    assert index_entry["primary_issue"] == "identity_theft"
    assert index_entry["account_number"] == {
        "per_bureau": {
            "transunion": "--",
            "experian": "--",
            "equifax": "****5678",
        },
        "consensus": "****5678",
    }
    assert index_entry["balance_owed"] == {
        "per_bureau": {
            "transunion": "--",
            "experian": "--",
            "equifax": "$75",
        }
    }

    assert result["packs_count"] == 1


def test_generate_frontend_packs_handles_missing_accounts(tmp_path):
    runs_root = tmp_path / "runs"
    sid = "S-empty"

    result = generate_frontend_packs_for_run(sid, runs_root=runs_root)

    index_path = runs_root / sid / "frontend" / "index.json"
    assert index_path.exists()
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == generator_module._FRONTEND_INDEX_SCHEMA_VERSION
    assert payload["accounts"] == []
    assert result["status"] == "success"
    assert result["packs_count"] == 0
    assert result["empty_ok"] is True
    assert result["built"] is True
    assert result["packs_dir"] == str(
        (runs_root / sid / "frontend" / "review" / "packs").absolute()
    )
    assert isinstance(result["last_built_at"], str)

    stage_index_path = runs_root / sid / "frontend" / "review" / "index.json"
    assert stage_index_path.exists()
    stage_payload = json.loads(stage_index_path.read_text(encoding="utf-8"))
    assert stage_payload["counts"]["packs"] == 0


def test_generate_frontend_packs_respects_feature_flag(tmp_path, monkeypatch):
    runs_root = tmp_path / "runs"
    sid = "S-disabled"

    monkeypatch.setenv("ENABLE_FRONTEND_PACKS", "0")

    result = generate_frontend_packs_for_run(sid, runs_root=runs_root)

    assert result == {
        "status": "skipped",
        "packs_count": 0,
        "empty_ok": True,
        "built": False,
        "packs_dir": str(
            (runs_root / sid / "frontend" / "review" / "packs").absolute()
        ),
        "last_built_at": None,
    }
    assert not (runs_root / sid).exists()


def test_generate_frontend_packs_task_exposed(tmp_path, monkeypatch):
    runs_root = tmp_path / "runs"
    sid = "S-task"

    monkeypatch.setenv("RUNS_ROOT", str(runs_root))

    import importlib
    import sys
    import types

    fake_requests = types.ModuleType("requests")
    monkeypatch.setitem(sys.modules, "requests", fake_requests)

    module_name = "backend.api.tasks"
    if module_name in sys.modules:
        api_tasks = importlib.reload(sys.modules[module_name])
    else:
        api_tasks = importlib.import_module(module_name)

    result = api_tasks.generate_frontend_packs_task.run(sid)

    assert result["packs_count"] == 0
    assert result["empty_ok"] is True

    index_path = runs_root / sid / "frontend" / "index.json"
    assert index_path.exists()


def test_generate_frontend_packs_continues_on_pack_write_failure(tmp_path, monkeypatch):
    runs_root = tmp_path / "runs"
    sid = "S-partial"

    account1_dir = runs_root / sid / "cases" / "accounts" / "1"
    account2_dir = runs_root / sid / "cases" / "accounts" / "2"

    shared_summary = {
        "account_id": "acct-success",
        "labels": {"creditor": "Cred", "account_type": "Loan", "status": "Open"},
    }
    shared_bureaus = {
        "transunion": {
            "account_number_display": "****9999",
            "balance_owed": "$50",
            "date_opened": "2023-01-01",
            "date_reported": "2023-02-01",
            "account_status": "Open",
            "account_type": "Loan",
        }
    }

    _write_json(account1_dir / "summary.json", {**shared_summary, "account_id": "acct-fail"})
    _write_json(account1_dir / "bureaus.json", shared_bureaus)
    _write_json(account1_dir / "tags.json", [{"kind": "issue", "type": "late_payment"}])

    _write_json(account2_dir / "summary.json", shared_summary)
    _write_json(account2_dir / "bureaus.json", shared_bureaus)
    _write_json(account2_dir / "tags.json", [{"kind": "issue", "type": "late_payment"}])

    original_write = generator_module._atomic_write_json

    def flaky_atomic_write(path, payload):
        if "acct-fail" in str(path):
            raise OSError("disk full")
        return original_write(path, payload)

    monkeypatch.setattr(generator_module, "_atomic_write_json", flaky_atomic_write)

    result = generator_module.generate_frontend_packs_for_run(sid, runs_root=runs_root)

    failing_pack = (
        runs_root / sid / "frontend" / "review" / "packs" / "acct-fail.json"
    )
    successful_pack = (
        runs_root / sid / "frontend" / "review" / "packs" / "acct-success.json"
    )

    assert not failing_pack.exists()
    assert successful_pack.exists()

    index_path = runs_root / sid / "frontend" / "index.json"
    index_payload = json.loads(index_path.read_text(encoding="utf-8"))

    assert index_payload["schema_version"] == generator_module._FRONTEND_INDEX_SCHEMA_VERSION
    assert index_payload["packs_count"] == 1
    index_entry = index_payload["accounts"][0]
    assert index_entry["account_id"] == "acct-success"
    assert index_entry["holder_name"] == ""
    assert index_entry["primary_issue"] == "late_payment"
    assert index_entry["account_number"] == {
        "per_bureau": {
            "transunion": "****9999",
            "experian": "--",
            "equifax": "--",
        },
        "consensus": "****9999",
    }
    assert index_entry["balance_owed"] == {
        "per_bureau": {
            "transunion": "$50",
            "experian": "--",
            "equifax": "--",
        }
    }

    assert result["status"] == "success"
    assert result["packs_count"] == 1
