import json
import unicodedata
import pytest
from pathlib import Path

import backend.ai.note_style_stage as note_style_stage_module
from backend.ai.note_style import prepare_and_send
from backend.ai.note_style_stage import (
    NoteStyleResponseAccount,
    build_note_style_pack_for_account,
    discover_note_style_response_accounts,
)
from backend.core.ai.paths import (
    ensure_note_style_account_paths,
    ensure_note_style_paths,
    note_style_result_filename,
)


_ZERO_WIDTH_TRANSLATION = {
    ord("\u200b"): " ",
    ord("\u200c"): " ",
    ord("\u200d"): " ",
    ord("\ufeff"): " ",
    ord("\u2060"): " ",
}


def _sanitize_note_text(note: str) -> str:
    normalized = unicodedata.normalize("NFKC", note)
    masked = note_style_stage_module._mask_contact_info(normalized)
    translated = masked.translate(_ZERO_WIDTH_TRANSLATION)
    return " ".join(translated.split()).strip()


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
    assert account_entry.result_filename == note_style_result_filename("Account 5!!")
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

    assert result["status"] == "skipped_low_signal"
    assert result["reason"] == "low_signal"
    assert "note_hash" not in result
    assert "prompt_salt" not in result

    paths = ensure_note_style_paths(runs_root, sid, create=False)
    account_paths = ensure_note_style_account_paths(paths, account_id, create=False)

    assert not account_paths.pack_file.exists()
    assert not account_paths.result_file.exists()

    index_payload = json.loads(paths.index_file.read_text(encoding="utf-8"))
    packs = index_payload["packs"]
    assert len(packs) == 1
    entry = packs[0]
    assert entry["status"] == "skipped_low_signal"
    assert "note_hash" not in entry


def test_note_style_stage_skips_when_result_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sid = "SID201"
    account_id = "acct-201"
    runs_root = tmp_path
    run_dir = runs_root / sid
    response_dir = run_dir / "frontend" / "review" / "responses"
    response_dir.mkdir(parents=True, exist_ok=True)

    account_dir = run_dir / "cases" / "accounts" / account_id
    account_dir.mkdir(parents=True, exist_ok=True)

    meta_payload = {"heading_guess": "Existing Creditor"}
    bureaus_payload = {
        "experian": {"account_type": "Credit Card", "reported_creditor": "Existing"}
    }
    tags_payload = [{"kind": "issue", "type": "wrong_amount"}]

    (account_dir / "meta.json").write_text(
        json.dumps(meta_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (account_dir / "bureaus.json").write_text(
        json.dumps(bureaus_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (account_dir / "tags.json").write_text(
        json.dumps(tags_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    note_text = "This account was already fixed."
    _write_response(
        response_dir / f"{account_id}.result.json",
        {"answers": {"explanation": note_text}},
    )

    paths = ensure_note_style_paths(runs_root, sid, create=True)
    account_paths = ensure_note_style_account_paths(paths, account_id, create=True)

    existing_result = {
        "sid": sid,
        "account_id": account_id,
        "analysis": {
            "tone": "confident",
            "context_hints": {"topic": "billing"},
            "emphasis": ["billing_error"],
            "confidence": 0.9,
            "risk_flags": [],
        },
    }
    account_paths.result_file.write_text(
        json.dumps(existing_result, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        note_style_stage_module.config,
        "NOTE_STYLE_SKIP_IF_RESULT_EXISTS",
        True,
    )

    outcome = build_note_style_pack_for_account(sid, account_id, runs_root=runs_root)

    assert outcome["status"] == "skipped_existing_analysis"
    assert outcome["reason"] == "existing_analysis"

    assert not account_paths.pack_file.exists()
    assert account_paths.result_file.read_text(encoding="utf-8").strip().endswith("}")

    index_payload = json.loads(paths.index_file.read_text(encoding="utf-8"))
    entry = index_payload["packs"][0]
    assert entry["account_id"] == account_id
    assert entry["status"] == "skipped_existing_analysis"

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
        },
    }
    meta_payload = {
        "account_id": account_id,
        "heading_guess": "Capital One",
        "issuer_slug": "capital-one",
        "account_number_tail": "1234",
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
        },
    )

    sanitized = _sanitize_note_text(note)

    result = build_note_style_pack_for_account(sid, account_id, runs_root=runs_root)
    assert result["status"] == "completed"
    assert "note_hash" not in result
    assert "prompt_salt" not in result

    paths = ensure_note_style_paths(runs_root, sid, create=False)
    account_paths = ensure_note_style_account_paths(paths, account_id, create=False)

    pack_payload = json.loads(account_paths.pack_file.read_text(encoding="utf-8"))
    result_payload = json.loads(account_paths.result_file.read_text(encoding="utf-8"))

    assert pack_payload["sid"] == sid
    assert pack_payload["account_id"] == account_id
    assert pack_payload["model"]
    assert pack_payload["messages"][0]["role"] == "system"
    assert "Prompt salt" not in pack_payload["messages"][0]["content"]

    user_message = pack_payload["messages"][1]
    assert user_message["role"] == "user"
    content = user_message["content"]
    assert content["note_text"] == sanitized
    assert content["account_display_name"] == "Capital One"
    assert content["primary_issue"] == "late_payment"
    assert "bureau_fields" in content
    assert content["bureau_fields"]["account_type"] == "Credit Card"
    assert content["bureau_fields"]["account_status"] == "Closed"
    assert "note_metrics" not in pack_payload
    assert "note_hash" not in pack_payload
    assert "prompt_salt" not in pack_payload
    assert "fingerprint_hash" not in pack_payload

    assert result_payload["sid"] == sid
    assert result_payload["account_id"] == account_id
    assert result_payload["note_metrics"]["char_len"] == len(sanitized)
    assert "account_context" not in result_payload
    assert "bureaus_summary" not in result_payload
    assert "note_hash" not in result_payload
    assert "prompt_salt" not in result_payload
    assert "fingerprint_hash" not in result_payload

    index_payload = json.loads(paths.index_file.read_text(encoding="utf-8"))
    entry = index_payload["packs"][0]
    assert entry["account_id"] == account_id
    assert entry["status"] == "built"
    assert "note_hash" not in entry
