from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
import unicodedata

import pytest

import backend.ai.note_style_stage as note_style_stage_module

from backend.ai.note_style import prepare_and_send
from backend.ai.note_style_results import store_note_style_result
from backend.ai.note_style_stage import (
    NoteStyleResponseAccount,
    build_note_style_pack_for_account,
    discover_note_style_response_accounts,
)
from backend.core.ai.paths import ensure_note_style_account_paths, ensure_note_style_paths
from backend.runflow.counters import note_style_stage_counts


_ZERO_WIDTH_TRANSLATION = {
    ord("\u200b"): " ",
    ord("\u200c"): " ",
    ord("\u200d"): " ",
    ord("\ufeff"): " ",
    ord("\u2060"): " ",
}


def _sanitize_note_text(note: str) -> str:
    normalized = unicodedata.normalize("NFKC", note)
    translated = normalized.translate(_ZERO_WIDTH_TRANSLATION)
    return " ".join(translated.split()).strip()


def _normalized_hash(text: str) -> str:
    normalized = " ".join(text.split()).strip().lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()



def _write_response(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def test_discover_note_style_accounts_empty(tmp_path: Path) -> None:
    sid = "SID100"
    results = discover_note_style_response_accounts(sid, runs_root=tmp_path)

    assert results == []


def test_discover_note_style_accounts_returns_sorted_entries(tmp_path: Path) -> None:
    sid = "SID101"
    responses_dir = tmp_path / sid / "frontend" / "review" / "responses"
    responses_dir.mkdir(parents=True, exist_ok=True)

    (responses_dir / "idx-000.result.json").touch()
    first = responses_dir / "idx-002.result.json"
    second = responses_dir / "Account 5!!.result.json"
    third = responses_dir / "idx-001.result.json"
    first_payload = {"answers": {"explain": "This is the second account."}}
    second_payload = {"note": "Customer provided additional context."}
    third_payload = {"answers": [{"note": "Short note"}]}
    for path, payload in zip(
        (first, second, third), (first_payload, second_payload, third_payload)
    ):
        _write_response(path, payload)

    # Write a file that should be ignored by discovery
    (responses_dir / "idx-003.summary.json").write_text("{}", encoding="utf-8")

    results = discover_note_style_response_accounts(sid, runs_root=tmp_path)

    assert [entry.account_id for entry in results] == [
        "Account 5!!",
        "idx-001",
        "idx-002",
    ]

    account_entry = results[0]
    assert isinstance(account_entry, NoteStyleResponseAccount)
    assert account_entry.normalized_account_id == "Account_5__"
    assert account_entry.pack_filename == "style_acc_Account_5__.jsonl"
    assert account_entry.result_filename == "acc_Account_5__.result.jsonl"
    assert account_entry.response_path == second.resolve()
    assert account_entry.response_relative.as_posix() == (
        f"runs/{sid}/frontend/review/responses/{second.name}"
    )


def test_discover_note_style_accounts_requires_note_field(tmp_path: Path) -> None:
    sid = "SID102"
    responses_dir = tmp_path / sid / "frontend" / "review" / "responses"
    responses_dir.mkdir(parents=True, exist_ok=True)

    first = responses_dir / "idx-000.result.json"
    second = responses_dir / "idx-001.result.json"
    missing = responses_dir / "idx-002.result.json"

    _write_response(first, {"note": ""})
    _write_response(second, {"answers": {"explain": "Customer provided a note."}})
    _write_response(missing, {"summary": "This is not a customer response."})

    results = discover_note_style_response_accounts(sid, runs_root=tmp_path)

    assert [entry.account_id for entry in results] == ["idx-000", "idx-001"]


def test_note_style_stage_skips_zero_length_response(tmp_path: Path) -> None:
    sid = "SID006"
    account_id = "idx-006"
    runs_root = tmp_path
    run_dir = runs_root / sid
    response_dir = run_dir / "frontend" / "review" / "responses"
    response_dir.mkdir(parents=True, exist_ok=True)

    response_path = response_dir / f"{account_id}.result.json"
    response_path.touch()

    result = build_note_style_pack_for_account(sid, account_id, runs_root=runs_root)
    expected_note_hash = hashlib.sha256(b"").hexdigest()

    assert result["status"] == "skipped_low_signal"
    assert result["reason"] == "low_signal"
    assert result["note_hash"] == expected_note_hash

    paths = ensure_note_style_paths(runs_root, sid, create=False)
    account_paths = ensure_note_style_account_paths(paths, account_id, create=False)

    assert not account_paths.pack_file.exists()
    assert not account_paths.result_file.exists()

    index_payload = json.loads(paths.index_file.read_text(encoding="utf-8"))
    packs = index_payload["packs"]
    assert len(packs) == 1
    entry = packs[0]
    assert entry["status"] == "skipped_low_signal"
    assert entry["note_hash"] == expected_note_hash

    runflow_payload = json.loads((run_dir / "runflow.json").read_text(encoding="utf-8"))
    note_style_stage = runflow_payload["stages"]["note_style"]
    assert note_style_stage["status"] == "success"
    assert note_style_stage["empty_ok"] is True


def test_prepare_and_send_without_responses_sets_empty_ok(tmp_path: Path) -> None:
    sid = "SIDEMPTY"

    result = prepare_and_send(sid, runs_root=tmp_path)

    assert result["accounts_discovered"] == 0
    assert result["packs_built"] == 0
    assert result["skipped"] == 0
    assert result["processed_accounts"] == []

    run_dir = tmp_path / sid
    runflow_payload = json.loads((run_dir / "runflow.json").read_text(encoding="utf-8"))
    stage_payload = runflow_payload["stages"]["note_style"]

    assert stage_payload["status"] == "success"
    assert stage_payload["empty_ok"] is True
    assert stage_payload["metrics"]["packs_total"] == 0
    assert stage_payload["results"]["results_total"] == 0
    assert stage_payload["results"]["completed"] == 0
    assert stage_payload["results"]["failed"] == 0

    summary = stage_payload["summary"]
    assert summary["empty_ok"] is True
    assert summary["packs_total"] == 0
    assert summary["results_total"] == 0
    assert summary["completed"] == 0
    assert summary["failed"] == 0

def test_note_style_stage_builds_artifacts(tmp_path: Path) -> None:
    sid = "SID001"
    account_id = "idx-001"
    runs_root = tmp_path
    run_dir = runs_root / sid
    response_dir = run_dir / "frontend" / "review" / "responses"
    note = "Please help, the bank made an error and I already paid this account."

    account_dir = run_dir / "cases" / "accounts" / "1"
    account_dir.mkdir(parents=True, exist_ok=True)
    summary_payload = {"account_id": account_id}
    bureaus_payload = {
        "experian": {
            "reported_creditor": "Capital One Bank",
            "account_type": "Credit Card",
            "account_status": "Closed",
            "payment_status": "Late 30 Days",
            "creditor_type": "Bank",
            "date_opened": "01/15/2020",
            "date_reported": "2024-03-10",
            "date_of_last_activity": "2024-02-05",
            "closed_date": "12/31/2023",
            "last_verified": "2024-03-01",
            "balance_owed": "$100",
            "high_balance": "$200",
            "past_due_amount": "$0",
            "account_number_display": "****1234",
            "creditor_name": "Capital One",
        },
        "equifax": {
            "reported_creditor": "Capital One Bank",
            "account_type": "Credit Card",
            "account_status": "Open",
            "payment_status": None,
            "creditor_type": "Bank",
            "date_opened": "2020-01-15",
            "date_reported": "2024-03-10",
            "date_of_last_activity": "2024-02-05",
            "closed_date": None,
            "last_verified": "03/01/2024",
            "balance_owed": "$100",
            "high_balance": "$250",
            "past_due_amount": "$25.00",
            "account_number_display": "****1234",
            "creditor_name": "Capital One",
        },
        "transunion": {
            "reported_creditor": "Capital One Bank",
            "account_type": "Credit Card",
            "account_status": "Closed",
            "payment_status": "Late 30 Days",
            "creditor_type": "Bank",
            "date_opened": "2020-01-15",
            "date_reported": "03/10/2024",
            "date_of_last_activity": "02/05/2024",
            "closed_date": "2023-12-31",
            "last_verified": "2024/03/01",
            "balance_owed": "$100.00",
            "high_balance": "$200",
            "past_due_amount": "$0",
            "account_number_display": "****1234",
            "creditor_name": "Capital One",
        },
    }
    meta_payload = {
        "account_id": account_id,
        "heading_guess": "Capital One",
        "issuer_slug": "capital-one",
        "account_number_tail": "1234",
        "bureau_presence": {
            "transunion": True,
            "experian": True,
            "equifax": False,
        },
    }
    tags_payload = [
        {"kind": "issue", "type": "late_payment"},
        {"kind": "note", "type": "call_customer"},
    ]

    (account_dir / "summary.json").write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (account_dir / "bureaus.json").write_text(
        json.dumps(bureaus_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (account_dir / "meta.json").write_text(
        json.dumps(meta_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (account_dir / "tags.json").write_text(
        json.dumps(tags_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    _write_response(
        response_dir / f"{account_id}.result.json",
        {
            "sid": sid,
            "account_id": account_id,
            "answers": {
                "explanation": note,
                "ui_allegations_selected": ["not_mine", "wrong_amount"],
            },
            "received_at": "2024-01-01T00:00:00Z",
        },
    )

    sanitized = _sanitize_note_text(note)
    expected_hash = _normalized_hash(sanitized)
    expected_note_hash = hashlib.sha256(note.encode("utf-8")).hexdigest()

    result = build_note_style_pack_for_account(sid, account_id, runs_root=runs_root)
    assert result["status"] == "completed"
    assert result["note_hash"] == expected_note_hash
    assert 8 <= len(result["prompt_salt"]) <= 12

    paths = ensure_note_style_paths(runs_root, sid, create=False)
    account_paths = ensure_note_style_account_paths(paths, account_id, create=False)

    assert account_paths.pack_file.is_file()
    assert account_paths.result_file.is_file()
    pack_payload = json.loads(account_paths.pack_file.read_text(encoding="utf-8"))
    result_payload = json.loads(account_paths.result_file.read_text(encoding="utf-8"))

    manifest_payload = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    note_style_manifest = (
        manifest_payload.get("ai", {})
        .get("packs", {})
        .get("note_style", {})
    )
    assert note_style_manifest["base"] == str(paths.base)
    assert note_style_manifest["packs_dir"] == str(paths.packs_dir)
    assert note_style_manifest["results_dir"] == str(paths.results_dir)
    assert note_style_manifest["index"] == str(paths.index_file)
    assert note_style_manifest["logs"].endswith("ai_packs/note_style/logs.txt")
    last_built_at = note_style_manifest.get("last_built_at")
    assert isinstance(last_built_at, str)
    status_manifest = note_style_manifest.get("status")
    assert isinstance(status_manifest, dict)
    assert status_manifest.get("built") is True
    assert status_manifest.get("completed_at") == last_built_at
    ai_status = (
        manifest_payload.get("ai", {})
        .get("status", {})
        .get("note_style", {})
    )
    assert ai_status.get("built") is True
    assert "sent" not in ai_status
    assert ai_status.get("completed_at") == last_built_at

    assert pack_payload["prompt_salt"] == result_payload["prompt_salt"] == result["prompt_salt"]
    pack_messages = pack_payload["messages"]
    assert isinstance(pack_messages, list)
    assert pack_messages[0]["role"] == "system"
    assert pack_messages[0]["content"].startswith(
        "You extract structured style from a customer's free-text note."
    )
    assert pack_messages[0]["content"].strip().endswith(
        f"Prompt salt: {result['prompt_salt']}"
    )
    assert pack_messages[1]["role"] == "user"
    user_payload = pack_messages[1]["content"]
    assert isinstance(user_payload, dict)
    metadata_payload = user_payload.get("metadata")
    assert isinstance(metadata_payload, dict)
    expected_context = {
        "identity": {
            "primary_issue": "late_payment",
            "account_id": account_id,
            "reported_creditor": "Capital One",
        },
        "meta": {
            "heading_guess": "Capital One",
            "issuer_slug": "capital-one",
            "account_number_tail": "1234",
            "bureau_presence": {
                "transunion": True,
                "experian": True,
                "equifax": False,
            },
        },
        "tags": {
            "issues": ["late_payment"],
            "other": ["note:call_customer"],
        },
        "bureaus": {
            "per_bureau": {
                "equifax": {
                    "reported_creditor": "Capital One Bank",
                    "account_type": "Credit Card",
                    "account_status": "Open",
                    "creditor_type": "Bank",
                    "date_opened": "2020-01-15",
                    "date_reported": "2024-03-10",
                    "date_of_last_activity": "2024-02-05",
                    "last_verified": "2024-03-01",
                    "balance_owed": "100",
                    "high_balance": "250",
                    "past_due_amount": "25",
                },
                "experian": {
                    "reported_creditor": "Capital One Bank",
                    "account_type": "Credit Card",
                    "account_status": "Closed",
                    "payment_status": "Late 30 Days",
                    "creditor_type": "Bank",
                    "date_opened": "2020-01-15",
                    "date_reported": "2024-03-10",
                    "date_of_last_activity": "2024-02-05",
                    "closed_date": "2023-12-31",
                    "last_verified": "2024-03-01",
                    "balance_owed": "100",
                    "high_balance": "200",
                    "past_due_amount": "0",
                },
                "transunion": {
                    "reported_creditor": "Capital One Bank",
                    "account_type": "Credit Card",
                    "account_status": "Closed",
                    "payment_status": "Late 30 Days",
                    "creditor_type": "Bank",
                    "date_opened": "2020-01-15",
                    "date_reported": "2024-03-10",
                    "date_of_last_activity": "2024-02-05",
                    "closed_date": "2023-12-31",
                    "last_verified": "2024-03-01",
                    "balance_owed": "100",
                    "high_balance": "200",
                    "past_due_amount": "0",
                },
            },
            "majority_values": {
                "reported_creditor": "Capital One Bank",
                "account_type": "Credit Card",
                "account_status": "Closed",
                "payment_status": "Late 30 Days",
                "creditor_type": "Bank",
                "date_opened": "2020-01-15",
                "date_reported": "2024-03-10",
                "date_of_last_activity": "2024-02-05",
                "closed_date": "2023-12-31",
                "last_verified": "2024-03-01",
                "balance_owed": "100",
                "high_balance": "200",
                "past_due_amount": "0",
            },
            "disagreements": {
                "account_status": {
                    "equifax": "Open",
                    "experian": "Closed",
                    "transunion": "Closed",
                },
                "high_balance": {
                    "equifax": "250",
                    "experian": "200",
                    "transunion": "200",
                },
                "past_due_amount": {
                    "equifax": "25",
                    "experian": "0",
                    "transunion": "0",
                },
            },
        },
    }
    expected_fingerprint = {
        "account_id": "idx-001",
        "identity": {
            "reported_creditor": "capital-one",
            "account_tail": "1234",
        },
        "core_issue": "late-payment",
        "financial": {
            "account_type": "credit-card",
            "account_status": "closed",
            "payment_status": "late-30-days",
        },
        "dates": {
            "opened": "2020-01-15",
            "last_activity": "2024-02-05",
        },
        "disagreements": True,
    }
    expected_fingerprint_hash = hashlib.sha256(
        json.dumps(
            expected_fingerprint,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    assert metadata_payload == {
        "sid": sid,
        "account_id": account_id,
        "fingerprint_hash": expected_fingerprint_hash,
        "channel": "frontend_review",
        "lang": "auto",
    }
    assert user_payload["note_text"] == note

    assert 8 <= len(pack_payload["prompt_salt"]) <= 12
    assert pack_payload["note_hash"] == expected_note_hash
    assert pack_payload["model"] == "gpt-4o-mini"
    assert (
        pack_payload["source_response_path"]
        == f"runs/{sid}/frontend/review/responses/{account_id}.result.json"
    )
    assert pack_payload["fingerprint"] == expected_fingerprint
    assert pack_payload["fingerprint_hash"] == expected_fingerprint_hash
    assert "account_context" not in pack_payload
    assert pack_payload["ui_allegations_selected"] == ["not_mine", "wrong_amount"]
    assert "extractor" not in pack_payload
    assert "analysis" not in result_payload
    assert "extractor" not in result_payload
    assert note not in account_paths.result_file.read_text(encoding="utf-8")
    assert result_payload["note_hash"] == expected_note_hash
    assert result_payload["source_hash"] == expected_hash
    assert result_payload["note_metrics"] == {
        "char_len": len(sanitized),
        "word_len": len(sanitized.split()),
    }
    assert result_payload["fingerprint"] == expected_fingerprint
    assert result_payload["fingerprint_hash"] == expected_fingerprint_hash
    assert "account_context" not in result_payload
    assert result_payload["ui_allegations_selected"] == ["not_mine", "wrong_amount"]

    index_payload = json.loads(paths.index_file.read_text(encoding="utf-8"))
    assert index_payload["schema_version"] == 1
    assert index_payload["root"] == "."
    assert index_payload["packs_dir"] == "packs"
    assert index_payload["results_dir"] == "results"
    packs = index_payload["packs"]
    assert len(packs) == 1
    first_entry = packs[0]
    assert first_entry["account_id"] == account_id
    assert first_entry["status"] == "built"
    expected_pack = account_paths.pack_file.relative_to(paths.base).as_posix()
    assert first_entry["pack"] == expected_pack
    assert "result" not in first_entry
    assert first_entry["note_hash"] == expected_note_hash
    assert first_entry["lines"] == 1
    assert first_entry["built_at"]

    runflow_payload = json.loads((run_dir / "runflow.json").read_text(encoding="utf-8"))
    note_style_stage = runflow_payload["stages"]["note_style"]
    assert note_style_stage["status"] == "built"
    assert note_style_stage["empty_ok"] is False
    assert note_style_stage["metrics"]["packs_total"] == 1
    assert note_style_stage["results"]["results_total"] == 1
    assert note_style_stage["results"]["completed"] == 0
    assert note_style_stage["results"]["failed"] == 0

    summary = note_style_stage["summary"]
    assert summary["empty_ok"] is False
    assert summary["results_total"] == 1
    assert summary["completed"] == 0
    assert summary["failed"] == 0
    assert summary["packs_total"] == 1
    assert summary["metrics"]["packs_total"] == 1
    assert summary["results"]["completed"] == 0


def test_note_style_identity_primary_issue_absent(tmp_path: Path) -> None:
    sid = "SIDPRI0"
    account_id = "acct-no-issue"
    runs_root = tmp_path
    run_dir = runs_root / sid
    account_dir = run_dir / "cases" / "accounts" / "10"
    response_dir = run_dir / "frontend" / "review" / "responses"
    note = "The report is inaccurate and needs review."

    summary_payload = {"account_id": account_id}
    bureaus_payload = {
        "transunion": {
            "reported_creditor": "Example Creditor",
            "account_status": "Open",
        }
    }
    meta_payload = {"account_id": account_id, "heading_guess": "Example Creditor"}
    tags_payload = [{"kind": "note", "type": "call_customer"}]

    account_dir.mkdir(parents=True, exist_ok=True)
    (account_dir / "summary.json").write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (account_dir / "bureaus.json").write_text(
        json.dumps(bureaus_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (account_dir / "meta.json").write_text(
        json.dumps(meta_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (account_dir / "tags.json").write_text(
        json.dumps(tags_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    _write_response(
        response_dir / f"{account_id}.result.json",
        {
            "sid": sid,
            "account_id": account_id,
            "answers": {"explanation": note},
        },
    )

    build_note_style_pack_for_account(sid, account_id, runs_root=runs_root)

    paths = ensure_note_style_paths(runs_root, sid, create=False)
    account_paths = ensure_note_style_account_paths(paths, account_id, create=False)
    pack_payload = json.loads(account_paths.pack_file.read_text(encoding="utf-8"))
    fingerprint = pack_payload["fingerprint"]
    identity = fingerprint["identity"]

    assert identity.get("primary_issue") is None
    assert fingerprint.get("account_id") == "acct-no-issue"
    assert identity.get("reported_creditor") == "example-creditor"


def test_note_style_stage_emits_structured_logs(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    sid = "SIDLOG1"
    account_id = "idx-log"
    runs_root = tmp_path
    response_dir = runs_root / sid / "frontend" / "review" / "responses"

    _write_response(
        response_dir / f"{account_id}.result.json",
        {
            "sid": sid,
            "account_id": account_id,
            "answers": {"explanation": "Bank error, already resolved."},
        },
    )

    caplog.set_level("INFO", logger="backend.ai.note_style_stage")
    result = build_note_style_pack_for_account(sid, account_id, runs_root=runs_root)

    messages = [
        record.getMessage()
        for record in caplog.records
        if record.name == "backend.ai.note_style_stage"
    ]

    assert any("NOTE_STYLE_DISCOVERY_DETAIL" in message for message in messages)
    assert any(
        "NOTE_STYLE_PACK_BUILT" in message
        and f"prompt_salt={result['prompt_salt']}" in message
        for message in messages
    )
    assert any(
        "NOTE_STYLE_INDEX_UPDATED" in message and "status=built" in message
        for message in messages
    )
    assert any("NOTE_STYLE_REFRESH" in message for message in messages)


def test_note_style_stage_refresh_promotes_on_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sid = "SID001A"
    account_id = "idx-001A"
    runs_root = tmp_path
    run_dir = runs_root / sid
    response_dir = run_dir / "frontend" / "review" / "responses"
    note = "We appreciate the help but still need assistance."

    response_path = response_dir / f"{account_id}.result.json"
    _write_response(
        response_path,
        {
            "sid": sid,
            "account_id": account_id,
            "answers": {"explanation": note},
        },
    )

    build_note_style_pack_for_account(sid, account_id, runs_root=runs_root)

    monkeypatch.setenv("RUNFLOW_EVENTS", "1")
    monkeypatch.setattr(
        "backend.ai.note_style_results.runflow_barriers_refresh", lambda _: None
    )
    monkeypatch.setattr(
        "backend.ai.note_style_results.reconcile_umbrella_barriers",
        lambda _sid, runs_root=None: {},
    )

    result_payload = {
        "sid": sid,
        "account_id": account_id,
        "analysis": {
            "tone": {"value": "supportive", "confidence": 0.9, "risk_flags": []},
            "context_hints": {
                "values": ["lender_fault"],
                "confidence": 0.8,
                "risk_flags": [],
            },
            "emphasis": {
                "values": ["help_request"],
                "confidence": 0.7,
                "risk_flags": [],
            },
        },
        "prompt_salt": "salt-value",
        "note_hash": "abc123",
    }

    store_note_style_result(
        sid,
        account_id,
        result_payload,
        runs_root=runs_root,
        completed_at="2024-02-10T00:00:00Z",
    )

    runflow_payload = json.loads((run_dir / "runflow.json").read_text(encoding="utf-8"))
    stage_payload = runflow_payload["stages"]["note_style"]
    assert stage_payload["status"] == "success"
    assert stage_payload["empty_ok"] is False
    assert stage_payload["metrics"]["packs_total"] == 1
    assert stage_payload["results"]["results_total"] == 1
    assert stage_payload["results"]["completed"] == 1
    assert stage_payload["results"]["failed"] == 0

    summary = stage_payload["summary"]
    assert summary["empty_ok"] is False
    assert summary["packs_total"] == 1
    assert summary["results_total"] == 1
    assert summary["completed"] == 1
    assert summary["failed"] == 0
    assert summary["metrics"]["packs_total"] == 1
    assert summary["results"]["completed"] == 1

    events_path = run_dir / "runflow_events.jsonl"
    assert events_path.exists()
    events = [
        json.loads(line)
        for line in events_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    refresh_events = [event for event in events if event.get("event") == "note_style_stage_refresh"]
    assert refresh_events, events
    latest_event = refresh_events[-1]
    assert latest_event["status"] == "success"
    assert latest_event["results_total"] == 1
    assert latest_event["results_completed"] == 1
    assert latest_event.get("results_failed", 0) == 0


def test_note_style_stage_minimal_smoke(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sid = "SID012"
    runs_root = tmp_path
    run_dir = runs_root / sid
    response_dir = run_dir / "frontend" / "review" / "responses"

    strong_account = "idx-strong"
    low_signal_account = "idx-low"
    casual_account = "idx-casual"

    strong_note = "This account is not mine. I never opened it."
    low_signal_note = "please fix"
    casual_note = "I paid this off last year, please remove it."

    _write_response(
        response_dir / f"{strong_account}.result.json",
        {
            "sid": sid,
            "account_id": strong_account,
            "answers": {"explanation": strong_note},
        },
    )
    _write_response(
        response_dir / f"{low_signal_account}.result.json",
        {
            "sid": sid,
            "account_id": low_signal_account,
            "answers": {"explanation": low_signal_note},
        },
    )
    _write_response(
        response_dir / f"{casual_account}.result.json",
        {
            "sid": sid,
            "account_id": casual_account,
            "answers": {"explanation": casual_note},
        },
    )

    strong_result = build_note_style_pack_for_account(
        sid, strong_account, runs_root=runs_root
    )
    low_signal_result = build_note_style_pack_for_account(
        sid, low_signal_account, runs_root=runs_root
    )
    casual_result = build_note_style_pack_for_account(
        sid, casual_account, runs_root=runs_root
    )

    assert strong_result["status"] == "completed"
    assert casual_result["status"] == "completed"
    assert low_signal_result["status"] == "skipped_low_signal"

    monkeypatch.setattr(
        "backend.ai.note_style_results.runflow_barriers_refresh", lambda _sid: None
    )
    monkeypatch.setattr(
        "backend.ai.note_style_results.reconcile_umbrella_barriers",
        lambda _sid, runs_root=None: {},
    )

    strong_source_hash = _normalized_hash(_sanitize_note_text(strong_note))
    casual_source_hash = _normalized_hash(_sanitize_note_text(casual_note))

    store_note_style_result(
        sid,
        strong_account,
        {
            "sid": sid,
            "account_id": strong_account,
            "analysis": {
                "tone": {"value": "assertive", "confidence": 0.8, "risk_flags": []},
                "context_hints": {
                    "timeframe": {"month": None, "relative": None},
                    "topic": "not_mine",
                    "entities": {"creditor": None, "amount": None},
                },
                "emphasis": ["ownership_dispute"],
                "confidence": 0.78,
                "risk_flags": [],
            },
            "prompt_salt": strong_result["prompt_salt"],
            "note_hash": strong_result["note_hash"],
            "source_hash": strong_source_hash,
        },
        runs_root=runs_root,
        completed_at="2024-03-01T00:00:00Z",
    )

    store_note_style_result(
        sid,
        casual_account,
        {
            "sid": sid,
            "account_id": casual_account,
            "analysis": {
                "tone": {"value": "conversational", "confidence": 0.7, "risk_flags": []},
                "context_hints": {
                    "timeframe": {"month": None, "relative": "last_year"},
                    "topic": "payment_dispute",
                    "entities": {"creditor": None, "amount": None},
                },
                "emphasis": ["paid_already", "update_requested"],
                "confidence": 0.74,
                "risk_flags": [],
            },
            "prompt_salt": casual_result["prompt_salt"],
            "note_hash": casual_result["note_hash"],
            "source_hash": casual_source_hash,
        },
        runs_root=runs_root,
        completed_at="2024-03-01T00:05:00Z",
    )

    paths = ensure_note_style_paths(runs_root, sid, create=False)
    index_payload = json.loads(paths.index_file.read_text(encoding="utf-8"))
    packs = sorted(index_payload["packs"], key=lambda item: item["account_id"])

    assert {entry["account_id"] for entry in packs} == {
        strong_account,
        low_signal_account,
        casual_account,
    }
    skipped_entry = next(
        entry for entry in packs if entry["account_id"] == low_signal_account
    )
    assert skipped_entry["status"] == "skipped_low_signal"

    runflow_payload = json.loads((run_dir / "runflow.json").read_text(encoding="utf-8"))
    note_style_stage = runflow_payload["stages"]["note_style"]
    assert note_style_stage["status"] == "success"
    assert note_style_stage["results"]["results_total"] == 2
    assert note_style_stage["results"]["completed"] == 2
    assert note_style_stage["summary"]["results_total"] == 2


def test_note_style_stage_idempotent_for_unchanged_response(tmp_path: Path) -> None:
    sid = "SID002"
    account_id = "idx-002"
    runs_root = tmp_path
    run_dir = runs_root / sid
    response_dir = run_dir / "frontend" / "review" / "responses"
    note = "Please help fix this error."

    response_path = response_dir / f"{account_id}.result.json"
    _write_response(
        response_path,
        {
            "sid": sid,
            "account_id": account_id,
            "answers": {"explanation": note},
            "received_at": "2024-01-02T00:00:00Z",
        },
    )

    first = build_note_style_pack_for_account(sid, account_id, runs_root=runs_root)
    assert first["status"] == "completed"

    paths = ensure_note_style_paths(runs_root, sid, create=False)
    account_paths = ensure_note_style_account_paths(paths, account_id, create=False)
    initial_pack = account_paths.pack_file.read_text(encoding="utf-8")
    initial_result = account_paths.result_file.read_text(encoding="utf-8")
    initial_index = paths.index_file.read_text(encoding="utf-8")

    second = build_note_style_pack_for_account(sid, account_id, runs_root=runs_root)
    assert second["status"] == "unchanged"
    assert account_paths.pack_file.read_text(encoding="utf-8") == initial_pack
    assert account_paths.result_file.read_text(encoding="utf-8") == initial_result
    assert paths.index_file.read_text(encoding="utf-8") == initial_index


def test_note_style_stage_updates_on_modified_note(tmp_path: Path) -> None:
    sid = "SID003"
    account_id = "idx-003"
    runs_root = tmp_path
    run_dir = runs_root / sid
    response_dir = run_dir / "frontend" / "review" / "responses"

    response_path = response_dir / f"{account_id}.result.json"
    original_note = "Please help correct this."
    _write_response(
        response_path,
        {
            "sid": sid,
            "account_id": account_id,
            "answers": {"explanation": original_note},
            "received_at": "2024-01-03T00:00:00Z",
        },
    )

    first = build_note_style_pack_for_account(sid, account_id, runs_root=runs_root)
    first_salt = first["prompt_salt"]
    first_hash = first["note_hash"]

    updated_note = "This is urgent and I dispute this account."
    _write_response(
        response_path,
        {
            "sid": sid,
            "account_id": account_id,
            "answers": {"explanation": updated_note},
            "received_at": "2024-01-03T00:10:00Z",
        },
    )

    updated = build_note_style_pack_for_account(sid, account_id, runs_root=runs_root)
    assert updated["status"] == "completed"
    assert updated["prompt_salt"] != first_salt
    assert updated["note_hash"] != first_hash

    paths = ensure_note_style_paths(runs_root, sid, create=False)
    account_paths = ensure_note_style_account_paths(paths, account_id, create=False)
    updated_payload = json.loads(account_paths.result_file.read_text(encoding="utf-8"))
    expected_hash = hashlib.sha256(updated_note.encode("utf-8")).hexdigest()
    expected_source = _normalized_hash(_sanitize_note_text(updated_note))
    assert updated_payload["note_hash"] == expected_hash
    assert updated_payload["source_hash"] == expected_source
    assert updated_payload["prompt_salt"] == updated["prompt_salt"]
    metrics = updated_payload["note_metrics"]
    sanitized = _sanitize_note_text(updated_note)
    assert metrics == {"char_len": len(sanitized), "word_len": len(sanitized.split())}

def test_note_style_stage_sanitizes_note_text(tmp_path: Path) -> None:
    sid = "SID004"
    account_id = "idx-004"
    runs_root = tmp_path
    run_dir = runs_root / sid
    response_dir = run_dir / "frontend" / "review" / "responses"
    note = "  Héllo\u00a0Bank\u2009\nAlready\u00a0paid  "

    _write_response(
        response_dir / f"{account_id}.result.json",
        {
            "sid": sid,
            "account_id": account_id,
            "answers": {"explanation": note},
            "received_at": "2024-01-04T00:00:00Z",
        },
    )

    sanitized = _sanitize_note_text(note)
    expected_hash = _normalized_hash(sanitized)
    expected_note_hash = hashlib.sha256(note.encode("utf-8")).hexdigest()

    result = build_note_style_pack_for_account(sid, account_id, runs_root=runs_root)
    assert result["status"] == "completed"
    assert result["note_hash"] == expected_note_hash

    paths = ensure_note_style_paths(runs_root, sid, create=False)
    account_paths = ensure_note_style_account_paths(paths, account_id, create=False)

    pack_payload = json.loads(account_paths.pack_file.read_text(encoding="utf-8"))
    result_payload = json.loads(account_paths.result_file.read_text(encoding="utf-8"))
    assert pack_payload["prompt_salt"] == result_payload["prompt_salt"] == result["prompt_salt"]

    pack_user_payload = pack_payload["messages"][1]["content"]
    expected_fingerprint = {
        "account_id": "idx-004",
        "disagreements": False,
    }
    expected_fingerprint_hash = hashlib.sha256(
        json.dumps(
            expected_fingerprint,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    assert pack_user_payload["note_text"] == note
    metadata_payload = pack_user_payload.get("metadata")
    assert metadata_payload == {
        "sid": sid,
        "account_id": account_id,
        "fingerprint_hash": expected_fingerprint_hash,
        "channel": "frontend_review",
        "lang": "auto",
    }

    assert pack_payload["note_hash"] == expected_note_hash
    assert pack_payload["fingerprint"] == expected_fingerprint
    assert pack_payload["fingerprint_hash"] == expected_fingerprint_hash
    assert "account_context" not in pack_payload
    assert result_payload["note_hash"] == expected_note_hash
    assert result_payload["source_hash"] == expected_hash
    assert result_payload["note_metrics"] == {
        "char_len": len(sanitized),
        "word_len": len(sanitized.split()),
    }
    assert result_payload["fingerprint"] == expected_fingerprint
    assert result_payload["fingerprint_hash"] == expected_fingerprint_hash
    assert "account_context" not in result_payload
    assert note not in account_paths.result_file.read_text(encoding="utf-8")


def test_note_style_stage_skips_when_note_sanitizes_empty(tmp_path: Path) -> None:
    sid = "SID005"
    account_id = "idx-005"
    runs_root = tmp_path
    run_dir = runs_root / sid
    response_dir = run_dir / "frontend" / "review" / "responses"
    note = "\u200b\n\t  "

    _write_response(
        response_dir / f"{account_id}.result.json",
        {
            "sid": sid,
            "account_id": account_id,
            "answers": {"explanation": note},
            "received_at": "2024-01-05T00:00:00Z",
        },
    )

    result = build_note_style_pack_for_account(sid, account_id, runs_root=runs_root)
    sanitized = _sanitize_note_text(note)
    expected_note_hash = hashlib.sha256(note.encode("utf-8")).hexdigest()
    assert result["status"] == "skipped_low_signal"
    assert result["reason"] == "low_signal"
    assert result["note_hash"] == expected_note_hash

    paths = ensure_note_style_paths(runs_root, sid, create=False)
    account_paths = ensure_note_style_account_paths(paths, account_id, create=False)

    assert not account_paths.pack_file.exists()
    assert not account_paths.result_file.exists()

    index_payload = json.loads(paths.index_file.read_text(encoding="utf-8"))
    packs = index_payload["packs"]
    assert len(packs) == 1
    entry = packs[0]
    assert entry["account_id"] == account_id
    assert entry["status"] == "skipped_low_signal"
    assert entry["pack"] == ""
    assert "result" not in entry
    assert entry["lines"] == 0
    assert entry["note_hash"] == expected_note_hash
    assert entry["built_at"]
    assert index_payload["root"] == "."

    runflow_payload = json.loads((run_dir / "runflow.json").read_text(encoding="utf-8"))
    note_style_stage = runflow_payload["stages"]["note_style"]
    assert note_style_stage["status"] == "success"
    assert note_style_stage["empty_ok"] is True
    assert note_style_stage["metrics"]["packs_total"] == 0
    assert note_style_stage["results"]["results_total"] == 0
    assert note_style_stage["results"]["completed"] == 0
    assert note_style_stage["results"]["failed"] == 0

    summary = note_style_stage["summary"]
    assert summary["empty_ok"] is True
    assert summary["packs_total"] == 0
    assert summary["results_total"] == 0
    assert summary["completed"] == 0
    assert summary["failed"] == 0
    assert summary["metrics"]["packs_total"] == 0
    assert summary["results"]["results_total"] == 0


def test_note_style_prompt_salt_varies_by_sid_and_account(tmp_path: Path) -> None:
    runs_root = tmp_path
    sid = "SID600"
    account_primary = "idx-600"
    account_secondary = "idx-601"
    sid_variant = "SID601"
    note = "Please help, I already paid this account in full."

    primary_response_dir = runs_root / sid / "frontend" / "review" / "responses"
    _write_response(
        primary_response_dir / f"{account_primary}.result.json",
        {
            "sid": sid,
            "account_id": account_primary,
            "answers": {"explanation": note},
        },
    )

    primary_result = build_note_style_pack_for_account(
        sid, account_primary, runs_root=runs_root
    )
    assert primary_result["status"] == "completed"
    primary_salt = primary_result["prompt_salt"]

    paths = ensure_note_style_paths(runs_root, sid, create=False)
    primary_account_paths = ensure_note_style_account_paths(
        paths, account_primary, create=False
    )
    payload = json.loads(primary_account_paths.result_file.read_text(encoding="utf-8"))
    assert payload["prompt_salt"] == primary_salt

    unchanged = build_note_style_pack_for_account(sid, account_primary, runs_root=runs_root)
    assert unchanged["status"] == "unchanged"
    payload_after = json.loads(
        primary_account_paths.result_file.read_text(encoding="utf-8")
    )
    assert payload_after["prompt_salt"] == primary_salt

    secondary_response_dir = primary_response_dir
    _write_response(
        secondary_response_dir / f"{account_secondary}.result.json",
        {
            "sid": sid,
            "account_id": account_secondary,
            "answers": {"explanation": note},
        },
    )

    secondary_result = build_note_style_pack_for_account(
        sid, account_secondary, runs_root=runs_root
    )
    assert secondary_result["status"] == "completed"
    secondary_salt = secondary_result["prompt_salt"]
    assert secondary_salt != primary_salt

    variant_response_dir = runs_root / sid_variant / "frontend" / "review" / "responses"
    _write_response(
        variant_response_dir / f"{account_primary}.result.json",
        {
            "sid": sid_variant,
            "account_id": account_primary,
            "answers": {"explanation": note},
        },
    )

    variant_result = build_note_style_pack_for_account(
        sid_variant, account_primary, runs_root=runs_root
    )
    assert variant_result["status"] == "completed"
    variant_salt = variant_result["prompt_salt"]
    assert variant_salt != primary_salt


def test_note_style_stage_skips_invalid_response_json(tmp_path: Path) -> None:
    sid = "SID700"
    account_id = "idx-700"
    runs_root = tmp_path
    run_dir = runs_root / sid
    response_dir = run_dir / "frontend" / "review" / "responses"
    response_path = response_dir / f"{account_id}.result.json"
    response_path.parent.mkdir(parents=True, exist_ok=True)
    response_path.write_text("{not-json", encoding="utf-8")

    result = build_note_style_pack_for_account(sid, account_id, runs_root=runs_root)
    assert result["status"] == "skipped"
    assert result["reason"] == "invalid_response"

    paths = ensure_note_style_paths(runs_root, sid, create=False)
    account_paths = ensure_note_style_account_paths(paths, account_id, create=False)
    assert not account_paths.pack_file.exists()
    assert not account_paths.result_file.exists()

    index_payload = json.loads(paths.index_file.read_text(encoding="utf-8"))
    assert index_payload["packs"] == []

    runflow_payload = json.loads((run_dir / "runflow.json").read_text(encoding="utf-8"))
    note_style_stage = runflow_payload["stages"]["note_style"]
    assert note_style_stage["status"] == "success"
    assert note_style_stage["empty_ok"] is True


def test_note_style_stage_ignores_summary_file(tmp_path: Path) -> None:
    sid = "SID710"
    account_id = "idx-710"
    runs_root = tmp_path
    response_dir = runs_root / sid / "frontend" / "review" / "responses"

    note = "Please help, I already paid this account."
    _write_response(
        response_dir / f"{account_id}.result.json",
        {
            "sid": sid,
            "account_id": account_id,
            "answers": {"explanation": note},
        },
    )

    summary_path = response_dir / f"{account_id}.summary.json"
    summary_payload = {
        "sid": sid,
        "account_id": account_id,
        "summary": {
            "answers": {"explanation": "This summary is urgent!!!"},
        },
    }
    summary_path.write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    result = build_note_style_pack_for_account(sid, account_id, runs_root=runs_root)
    assert result["status"] == "completed"

    paths = ensure_note_style_paths(runs_root, sid, create=False)
    account_paths = ensure_note_style_account_paths(paths, account_id, create=False)
    result_payload = json.loads(account_paths.result_file.read_text(encoding="utf-8"))
    assert "analysis" not in result_payload
    sanitized = _sanitize_note_text(note)
    assert result_payload["note_metrics"] == {"char_len": len(sanitized), "word_len": len(sanitized.split())}

    assert summary_path.read_text(encoding="utf-8") == json.dumps(
        summary_payload, ensure_ascii=False, indent=2
    )


def test_note_style_stage_counts_track_completion(tmp_path: Path) -> None:
    sid = "SID720"
    account_id = "idx-720"
    runs_root = tmp_path
    run_dir = runs_root / sid
    response_dir = run_dir / "frontend" / "review" / "responses"

    _write_response(
        response_dir / f"{account_id}.result.json",
        {
            "sid": sid,
            "account_id": account_id,
            "answers": {"explanation": "Already resolved with the bank."},
        },
    )

    result = build_note_style_pack_for_account(sid, account_id, runs_root=runs_root)
    assert result["status"] == "completed"

    counts_pending = note_style_stage_counts(run_dir)
    assert counts_pending == {"packs_total": 1, "packs_completed": 0, "packs_failed": 0}

    paths = ensure_note_style_paths(runs_root, sid, create=False)
    account_paths = ensure_note_style_account_paths(paths, account_id, create=False)
    payload = json.loads(account_paths.result_file.read_text(encoding="utf-8"))

    store_note_style_result(sid, account_id, payload, runs_root=runs_root)

    counts_completed = note_style_stage_counts(run_dir)
    assert counts_completed == {"packs_total": 1, "packs_completed": 1, "packs_failed": 0}


def test_build_note_style_pack_logs_missing_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    sid = "SID777"
    account_id = "idx-777"
    runs_root = tmp_path / "runs"
    run_dir = runs_root / sid
    account_dir = run_dir / "cases" / "accounts" / "0"
    account_dir.mkdir(parents=True, exist_ok=True)

    (account_dir / "meta.json").write_text("{}", encoding="utf-8")
    (account_dir / "bureaus.json").write_text("{}", encoding="utf-8")
    (account_dir / "tags.json").write_text("[]", encoding="utf-8")

    response_dir = run_dir / "frontend" / "review" / "responses"
    _write_response(
        response_dir / f"{account_id}.result.json",
        {
            "sid": sid,
            "account_id": account_id,
            "answers": {
                "explanation": "Bank error remains unresolved despite multiple payments."
            },
        },
    )

    original_write = note_style_stage_module._write_jsonl

    def _fake_write_jsonl(path: Path, row: dict[str, object]) -> None:
        if path.name.endswith(".result.jsonl"):
            return
        original_write(path, row)

    monkeypatch.setattr(note_style_stage_module, "_write_jsonl", _fake_write_jsonl)

    caplog.set_level("WARNING", logger="backend.ai.note_style_stage")

    build_note_style_pack_for_account(sid, account_id, runs_root=runs_root)

    paths = ensure_note_style_paths(runs_root, sid, create=False)
    log_path = paths.log_file
    assert log_path.exists()
    log_contents = log_path.read_text(encoding="utf-8")
    assert "missing_artifacts=result" in log_contents
    assert account_id in log_contents

    warning_messages = [
        record.getMessage()
        for record in caplog.records
        if record.name == "backend.ai.note_style_stage"
    ]
    assert any(
        "NOTE_STYLE_ARTIFACT_VALIDATION_FAILED" in message
        for message in warning_messages
    )
