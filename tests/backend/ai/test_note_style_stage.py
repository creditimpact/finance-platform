import json
import logging
from pathlib import Path

import pytest

from backend.ai.note_style import prepare_and_send
from backend.ai.note_style_stage import (
    NoteStyleResponseAccount,
    build_note_style_pack_for_account,
    discover_note_style_response_accounts,
)
from backend.core.ai.paths import ensure_note_style_account_paths, ensure_note_style_paths


def _write_response(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_manifest(run_dir: Path, account_id: str, *, account_dir: Path | None = None) -> Path:
    account_dir = account_dir or (run_dir / "cases" / "accounts" / account_id)
    try:
        dir_value = account_dir.relative_to(run_dir)
        dir_entry = dir_value.as_posix()
    except ValueError:
        dir_entry = str(account_dir)

    manifest_payload = {
        "artifacts": {
            "cases": {
                "accounts": {
                    account_id: {
                        "dir": dir_entry,
                        "meta": "meta.json",
                        "bureaus": "bureaus.json",
                        "tags": "tags.json",
                    }
                }
            }
        }
    }

    manifest_path = run_dir / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return account_dir


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
    assert account_entry.pack_filename == "acc_Account_5__.jsonl"
    assert account_entry.result_filename.endswith(".result.jsonl")
    assert account_entry.response_path == second.resolve()
    assert account_entry.response_relative.as_posix() == (
        f"{sid}/frontend/review/responses/{second.name}"
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

    assert [entry.account_id for entry in results] == ["idx-001"]


def test_note_style_stage_skips_missing_response(tmp_path: Path) -> None:
    sid = "SID006"
    account_id = "idx-006"
    run_dir = tmp_path / sid
    response_dir = run_dir / "frontend" / "review" / "responses"
    response_dir.mkdir(parents=True, exist_ok=True)

    response_path = response_dir / f"{account_id}.result.json"
    response_path.touch()

    account_dir = run_dir / "cases" / "accounts" / account_id
    account_dir.mkdir(parents=True, exist_ok=True)
    _write_manifest(run_dir, account_id, account_dir=account_dir)

    result = build_note_style_pack_for_account(sid, account_id, runs_root=tmp_path)

    assert result == {"status": "skipped", "reason": "no_response"}

    paths = ensure_note_style_paths(tmp_path, sid, create=False)
    account_paths = ensure_note_style_account_paths(paths, account_id, create=False)

    assert not account_paths.pack_file.exists()
    assert not account_paths.result_file.exists()
    assert not account_paths.debug_file.exists()


def test_note_style_stage_skips_when_note_missing(tmp_path: Path) -> None:
    sid = "SID007"
    account_id = "idx-007"
    run_dir = tmp_path / sid
    response_dir = run_dir / "frontend" / "review" / "responses"
    response_dir.mkdir(parents=True, exist_ok=True)

    _write_response(response_dir / f"{account_id}.result.json", {"note": "   "})

    account_dir = run_dir / "cases" / "accounts" / account_id
    account_dir.mkdir(parents=True, exist_ok=True)
    _write_manifest(run_dir, account_id, account_dir=account_dir)

    result = build_note_style_pack_for_account(sid, account_id, runs_root=tmp_path)

    assert result == {"status": "skipped", "reason": "no_note"}

    paths = ensure_note_style_paths(tmp_path, sid, create=False)
    account_paths = ensure_note_style_account_paths(paths, account_id, create=False)

    assert not account_paths.pack_file.exists()
    assert not account_paths.result_file.exists()


def test_note_style_stage_overwrites_existing_result(tmp_path: Path) -> None:
    sid = "SID201"
    account_id = "acct-201"
    run_dir = tmp_path / sid
    response_dir = run_dir / "frontend" / "review" / "responses"
    response_dir.mkdir(parents=True, exist_ok=True)

    account_dir = run_dir / "cases" / "accounts" / account_id
    account_dir.mkdir(parents=True, exist_ok=True)

    _write_manifest(run_dir, account_id, account_dir=account_dir)

    _write_response(
        response_dir / f"{account_id}.result.json",
        {"answers": {"explanation": "Existing note"}},
    )

    paths = ensure_note_style_paths(tmp_path, sid, create=True)
    account_paths = ensure_note_style_account_paths(paths, account_id, create=True)

    account_paths.result_file.write_text(
        json.dumps({"sid": sid, "account_id": account_id, "analysis": {"tone": "calm"}})
        + "\n",
        encoding="utf-8",
    )

    outcome = build_note_style_pack_for_account(sid, account_id, runs_root=tmp_path)

    assert outcome["status"] == "completed"

    assert not account_paths.result_file.exists()


def test_note_style_stage_builds_artifacts(tmp_path: Path) -> None:
    sid = "SID001"
    account_id = "idx-001"
    run_dir = tmp_path / sid
    response_dir = run_dir / "frontend" / "review" / "responses"
    note = "Please help, the bank made an error and I already paid this account."

    account_dir = run_dir / "cases" / "accounts" / account_id
    account_dir.mkdir(parents=True, exist_ok=True)
    meta_payload = {"heading_guess": "Capital One"}
    bureaus_payload = {
        "experian": {
            "reported_creditor": "Capital One Bank",
            "account_type": "Credit Card",
            "account_status": "Closed",
        }
    }
    tags_payload = [{"kind": "issue", "type": "late_payment"}]

    (account_dir / "meta.json").write_text(
        json.dumps(meta_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (account_dir / "bureaus.json").write_text(
        json.dumps(bureaus_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (account_dir / "tags.json").write_text(
        json.dumps(tags_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    _write_manifest(run_dir, account_id, account_dir=account_dir)

    _write_response(
        response_dir / f"{account_id}.result.json",
        {"answers": {"explanation": note}},
    )

    result = build_note_style_pack_for_account(sid, account_id, runs_root=tmp_path)

    assert result["status"] == "completed"

    paths = ensure_note_style_paths(tmp_path, sid, create=False)
    account_paths = ensure_note_style_account_paths(paths, account_id, create=False)

    pack_line = account_paths.pack_file.read_text(encoding="utf-8")
    pack_payload = json.loads(pack_line)

    assert not account_paths.result_file.exists()
    assert not account_paths.debug_file.exists()

    assert set(pack_payload.keys()) == {
        "meta_name",
        "primary_issue_tag",
        "bureau_data",
        "note_text",
        "messages",
    }
    assert pack_payload["meta_name"] == "Capital One"
    assert pack_payload["note_text"].startswith("Please help")
    bureau_data = pack_payload["bureau_data"]
    expected_bureau_fields = {
        "account_type",
        "account_status",
        "payment_status",
        "creditor_type",
        "date_opened",
        "date_reported",
        "date_of_last_activity",
        "closed_date",
        "last_verified",
        "balance_owed",
        "high_balance",
        "past_due_amount",
    }
    assert set(bureau_data.keys()) == expected_bureau_fields
    assert bureau_data["account_type"] == "Credit Card"
    assert bureau_data["account_status"] == "Closed"
    assert pack_payload["primary_issue_tag"] == "late_payment"
    assert pack_payload["messages"][0]["role"] == "system"

    user_message = pack_payload["messages"][1]
    assert user_message["role"] == "user"
    user_content = user_message["content"]
    assert isinstance(user_content, str)
    parsed_user_content = json.loads(user_content)
    assert parsed_user_content == {
        "meta_name": "Capital One",
        "primary_issue_tag": "late_payment",
        "bureau_data": bureau_data,
        "note_text": pack_payload["note_text"],
    }

    # The pack line should not contain deep context artifacts.
    assert '"bureaus"' not in pack_line
    assert '"tags"' not in pack_line
    assert '"experian"' not in pack_line
    assert '"transunion"' not in pack_line
    assert '"equifax"' not in pack_line


def test_note_style_manifest_registered_before_pack_build(tmp_path: Path) -> None:
    sid = "SIDREG"
    account_id = "acct-500"
    run_dir = tmp_path / sid
    response_dir = run_dir / "frontend" / "review" / "responses"

    account_dir = run_dir / "cases" / "accounts" / account_id
    account_dir.mkdir(parents=True, exist_ok=True)
    (account_dir / "meta.json").write_text(
        json.dumps({"heading_guess": "Example Creditor"}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (account_dir / "bureaus.json").write_text(
        json.dumps({"experian": {"reported_creditor": "Example Creditor"}}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (account_dir / "tags.json").write_text(
        json.dumps([{"kind": "issue", "type": "verification"}], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    _write_manifest(run_dir, account_id, account_dir=account_dir)

    _write_response(
        response_dir / f"{account_id}.result.json",
        {"answers": {"explanation": "Customer already resolved the balance."}},
    )

    result = build_note_style_pack_for_account(sid, account_id, runs_root=tmp_path)
    assert result["status"] == "completed"

    manifest_path = run_dir / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    note_style_section = (
        payload.get("ai", {})
        .get("packs", {})
        .get("note_style", {})
    )

    assert note_style_section, "note_style manifest section should be populated"

    paths = ensure_note_style_paths(tmp_path, sid, create=False)

    assert note_style_section["base"] == str(paths.base)
    assert note_style_section["dir"] == str(paths.base)
    assert note_style_section["packs"] == str(paths.packs_dir)
    assert note_style_section["packs_dir"] == str(paths.packs_dir)
    assert note_style_section["results"] == str(paths.results_dir)
    assert note_style_section["results_dir"] == str(paths.results_dir)
    assert note_style_section["index"] == str(paths.index_file)
    assert note_style_section["logs"] == str(paths.log_file)

    status_payload = note_style_section.get("status")
    assert isinstance(status_payload, dict)
    assert set(status_payload.keys()) >= {"built", "sent", "completed_at"}
    assert status_payload["built"] is True
    assert status_payload["sent"] is False
    assert status_payload["completed_at"] is None


def test_note_style_stage_marks_built_after_all_packs(tmp_path: Path) -> None:
    sid = "SID-BUILT-GATE"
    accounts = ["acct-1", "acct-2", "acct-3"]
    run_dir = tmp_path / sid
    responses_dir = run_dir / "frontend" / "review" / "responses"

    manifest_accounts: dict[str, dict[str, str]] = {}

    for account in accounts:
        account_dir = run_dir / "cases" / "accounts" / account
        account_dir.mkdir(parents=True, exist_ok=True)

        meta_payload = {"heading_guess": f"Creditor {account}"}
        bureaus_payload = {
            "experian": {"reported_creditor": f"Creditor {account}"}
        }
        tags_payload = [{"kind": "issue", "type": "dispute"}]

        (account_dir / "meta.json").write_text(
            json.dumps(meta_payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (account_dir / "bureaus.json").write_text(
            json.dumps(bureaus_payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (account_dir / "tags.json").write_text(
            json.dumps(tags_payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        manifest_accounts[account] = {
            "dir": f"cases/accounts/{account}",
            "meta": "meta.json",
            "bureaus": "bureaus.json",
            "tags": "tags.json",
        }

        _write_response(
            responses_dir / f"{account}.result.json",
            {"answers": {"explain": f"Note for {account}"}},
        )

    manifest_payload = {
        "artifacts": {"cases": {"accounts": manifest_accounts}},
    }

    manifest_path = run_dir / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    for account in accounts[:2]:
        result = build_note_style_pack_for_account(sid, account, runs_root=tmp_path)
        assert result["status"] == "completed"

    runflow_path = run_dir / "runflow.json"
    stage_payload = json.loads(runflow_path.read_text(encoding="utf-8"))["stages"][
        "note_style"
    ]
    assert stage_payload["status"] == "pending"
    metrics_payload = stage_payload.get("metrics") or {}
    assert metrics_payload.get("packs_total") == len(accounts)

    result = build_note_style_pack_for_account(sid, accounts[2], runs_root=tmp_path)
    assert result["status"] == "completed"

    stage_payload = json.loads(runflow_path.read_text(encoding="utf-8"))["stages"][
        "note_style"
    ]
    assert stage_payload["status"] == "built"
    metrics_payload = stage_payload.get("metrics") or {}
    assert metrics_payload.get("packs_total") == len(accounts)

def test_note_style_stage_uses_manifest_account_paths(tmp_path: Path) -> None:
    sid = "SIDMAN"
    account_id = "idx-007"
    run_dir = tmp_path / sid
    response_dir = run_dir / "frontend" / "review" / "responses"

    note = "Customer is disputing the reported balance."

    account_dir = run_dir / "cases" / "accounts" / "7"
    account_dir.mkdir(parents=True, exist_ok=True)

    meta_payload = {"heading_guess": "Sample Creditor", "account_name": "Sample Account"}
    bureaus_payload = {
        "transunion": {
            "reported_creditor": "Sample Creditor",
            "account_type": "Credit Card",
        }
    }
    tags_payload = [{"kind": "issue", "type": "balance_dispute"}]

    (account_dir / "meta.json").write_text(
        json.dumps(meta_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (account_dir / "bureaus.json").write_text(
        json.dumps(bureaus_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (account_dir / "tags.json").write_text(
        json.dumps(tags_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    manifest_payload = {
        "sid": sid,
        "artifacts": {
            "cases": {
                "accounts": {
                    account_id: {
                        "dir": str(account_dir),
                        "meta": str(account_dir / "meta.json"),
                        "bureaus": str(account_dir / "bureaus.json"),
                        "tags": str(account_dir / "tags.json"),
                    }
                }
            }
        },
    }

    (run_dir / "manifest.json").write_text(
        json.dumps(manifest_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    _write_response(
        response_dir / f"{account_id}.result.json",
        {"answers": {"explanation": note}},
    )

    result = build_note_style_pack_for_account(sid, account_id, runs_root=tmp_path)

    assert result["status"] == "completed"

    paths = ensure_note_style_paths(tmp_path, sid, create=False)
    account_paths = ensure_note_style_account_paths(paths, account_id, create=False)

    pack_payload = json.loads(account_paths.pack_file.read_text(encoding="utf-8"))

    assert pack_payload["meta_name"] == "Sample Creditor"
    bureau_data = pack_payload["bureau_data"]
    assert bureau_data["account_type"] == "Credit Card"
    assert pack_payload["primary_issue_tag"] == "balance_dispute"
    user_content = pack_payload["messages"][1]["content"]
    assert isinstance(user_content, str)
    parsed_content = json.loads(user_content)
    assert parsed_content == {
        "meta_name": "Sample Creditor",
        "primary_issue_tag": "balance_dispute",
        "bureau_data": bureau_data,
        "note_text": pack_payload["note_text"],
    }


def test_note_style_stage_resolves_relative_manifest_paths(tmp_path: Path) -> None:
    sid = "SIDREL"
    account_id = "idx-010"
    run_dir = tmp_path / sid
    response_dir = run_dir / "frontend" / "review" / "responses"

    note = "Relative manifest paths should resolve to account artifacts."

    account_dir = run_dir / "cases" / "accounts" / account_id
    account_dir.mkdir(parents=True, exist_ok=True)

    meta_payload = {"heading_guess": "Relative Creditor"}
    bureaus_payload = {"experian": {"account_status": "Open", "account_type": "Loan"}}
    tags_payload = [{"kind": "issue", "type": "relative_path"}]

    (account_dir / "meta.json").write_text(
        json.dumps(meta_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (account_dir / "bureaus.json").write_text(
        json.dumps(bureaus_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (account_dir / "tags.json").write_text(
        json.dumps(tags_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    manifest_payload = {
        "sid": sid,
        "artifacts": {
            "cases": {
                "accounts": {
                    account_id: {
                        "dir": "cases/accounts/idx-010",
                        "meta": "meta.json",
                        "bureaus": "cases/accounts/idx-010/bureaus.json",
                        "tags": "tags.json",
                    }
                }
            }
        },
    }

    (run_dir / "manifest.json").write_text(
        json.dumps(manifest_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    _write_response(
        response_dir / f"{account_id}.result.json",
        {"answers": {"explanation": note}},
    )

    result = build_note_style_pack_for_account(sid, account_id, runs_root=tmp_path)

    assert result["status"] == "completed"

    paths = ensure_note_style_paths(tmp_path, sid, create=False)
    account_paths = ensure_note_style_account_paths(paths, account_id, create=False)

    pack_payload = json.loads(account_paths.pack_file.read_text(encoding="utf-8"))

    assert pack_payload["meta_name"] == "Relative Creditor"
    bureau_data = pack_payload["bureau_data"]
    assert bureau_data["account_status"] == "Open"
    assert pack_payload["primary_issue_tag"] == "relative_path"
    user_content = pack_payload["messages"][1]["content"]
    assert isinstance(user_content, str)
    parsed_content = json.loads(user_content)
    assert parsed_content == {
        "meta_name": "Relative Creditor",
        "primary_issue_tag": "relative_path",
        "bureau_data": bureau_data,
        "note_text": pack_payload["note_text"],
    }


def test_note_style_stage_handles_missing_context(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    sid = "SID002"
    account_id = "idx-002"
    run_dir = tmp_path / sid
    response_dir = run_dir / "frontend" / "review" / "responses"
    note = "No context files exist for this account."

    account_dir = run_dir / "cases" / "accounts" / account_id
    account_dir.mkdir(parents=True, exist_ok=True)
    _write_manifest(run_dir, account_id, account_dir=account_dir)

    _write_response(
        response_dir / f"{account_id}.result.json",
        {"answers": {"explanation": note}},
    )

    with caplog.at_level(logging.WARNING, logger="backend.ai.note_style_stage"):
        result = build_note_style_pack_for_account(sid, account_id, runs_root=tmp_path)

    assert result["status"] == "completed"

    paths = ensure_note_style_paths(tmp_path, sid, create=False)
    account_paths = ensure_note_style_account_paths(paths, account_id, create=False)

    pack_payload = json.loads(account_paths.pack_file.read_text(encoding="utf-8"))

    assert not account_paths.result_file.exists()
    assert not account_paths.debug_file.exists()

    assert set(pack_payload.keys()) == {
        "meta_name",
        "primary_issue_tag",
        "bureau_data",
        "note_text",
        "messages",
    }
    assert pack_payload["meta_name"] == account_id
    bureau_data = pack_payload["bureau_data"]
    assert set(bureau_data.keys()) == {
        "account_type",
        "account_status",
        "payment_status",
        "creditor_type",
        "date_opened",
        "date_reported",
        "date_of_last_activity",
        "closed_date",
        "last_verified",
        "balance_owed",
        "high_balance",
        "past_due_amount",
    }
    assert all(value == "--" for value in bureau_data.values())
    assert pack_payload["primary_issue_tag"] is None

    assert "NOTE_STYLE_WARN: missing context for account idx-002 (meta/tags/bureaus)" in caplog.text

    user_context = pack_payload["messages"][1]["content"]
    assert isinstance(user_context, str)
    parsed_user_context = json.loads(user_context)
    assert parsed_user_context == {
        "meta_name": account_id,
        "primary_issue_tag": None,
        "bureau_data": bureau_data,
        "note_text": pack_payload["note_text"],
    }

    # Debug context snapshots should never leak into the pack payload that will be
    # forwarded to the AI model.
    assert "debug" not in pack_payload


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

    assert stage_payload["status"] == "empty"
    assert stage_payload["empty_ok"] is True
    assert stage_payload["metrics"]["packs_total"] == 0
    assert stage_payload["results"]["results_total"] == 0
    assert stage_payload["results"]["completed"] == 0
    assert stage_payload["results"]["failed"] == 0

    summary = stage_payload["summary"]
    assert summary["empty_ok"] is True
    assert summary["packs_total"] == 0
    assert summary["results_total"] == 0
