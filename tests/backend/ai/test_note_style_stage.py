from __future__ import annotations

import hashlib
import hmac
import json
from pathlib import Path
import unicodedata

from backend.ai.note_style_stage import build_note_style_pack_for_account
from backend.core.ai.paths import ensure_note_style_account_paths, ensure_note_style_paths


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


def _expected_salt(sid: str, account_id: str, note: str) -> str:
    sanitized_note = _sanitize_note_text(note)
    digest = _normalized_hash(sanitized_note)
    pepper = b"tests-note-style-pepper"
    message = f"{sid}:{account_id}:{digest}".encode("utf-8")
    return hmac.new(pepper, message, hashlib.sha256).hexdigest()[:16]


def _write_response(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def test_note_style_stage_builds_artifacts(tmp_path: Path) -> None:
    sid = "SID001"
    account_id = "idx-001"
    runs_root = tmp_path
    run_dir = runs_root / sid
    response_dir = run_dir / "frontend" / "review" / "responses"
    note = "Please help, the bank made an error and I already paid this account."

    _write_response(
        response_dir / f"{account_id}.result.json",
        {
            "sid": sid,
            "account_id": account_id,
            "answers": {"explanation": note},
            "received_at": "2024-01-01T00:00:00Z",
        },
    )

    result = build_note_style_pack_for_account(sid, account_id, runs_root=runs_root)
    assert result["status"] == "completed"

    paths = ensure_note_style_paths(runs_root, sid, create=False)
    account_paths = ensure_note_style_account_paths(paths, account_id, create=False)

    assert account_paths.pack_file.is_file()
    assert account_paths.result_file.is_file()
    pack_payload = json.loads(account_paths.pack_file.read_text(encoding="utf-8"))
    result_payload = json.loads(account_paths.result_file.read_text(encoding="utf-8"))

    sanitized = _sanitize_note_text(note)
    expected_hash = _normalized_hash(sanitized)
    expected_short_hash = expected_hash[:12]

    pack_messages = pack_payload["messages"]
    assert isinstance(pack_messages, list)
    assert pack_messages[0]["role"] == "system"
    assert "tone" in pack_messages[0]["content"].lower()
    assert pack_messages[1]["role"] == "user"
    assert pack_messages[1]["content"] == sanitized

    expected_salt = _expected_salt(sid, account_id, note)
    assert pack_payload["prompt_salt"] == expected_salt
    assert expected_salt in pack_messages[0]["content"]
    assert result_payload["prompt_salt"] == expected_salt
    assert pack_payload["note_hash"] == expected_short_hash
    assert pack_payload["model"] == "gpt-4o-mini"
    assert (
        pack_payload["source_response_path"]
        == f"runs/{sid}/frontend/review/responses/{account_id}.result.json"
    )
    assert sanitized not in account_paths.result_file.read_text(encoding="utf-8")
    assert result_payload["note_hash"] == expected_short_hash
    assert result_payload["source_hash"] == expected_hash
    assert result_payload["analysis"]["tone"]["value"] == "conciliatory"
    context_values = result_payload["analysis"]["context_hints"]["values"]
    assert "lender_fault" in context_values
    assert "payment_dispute" in context_values
    emphasis_values = result_payload["analysis"]["emphasis"]["values"]
    assert "support_request" in emphasis_values

    index_payload = json.loads(paths.index_file.read_text(encoding="utf-8"))
    totals = index_payload["totals"]
    assert totals["total"] == 1
    assert totals["completed"] == 1
    assert totals["failed"] == 0
    first_entry = index_payload["items"][0]
    assert first_entry["prompt_salt"] == expected_salt
    assert first_entry["source_hash"] == expected_hash
    assert first_entry["note_hash"] == expected_short_hash

    runflow_payload = json.loads((run_dir / "runflow.json").read_text(encoding="utf-8"))
    note_style_stage = runflow_payload["stages"]["note_style"]
    assert note_style_stage["status"] == "success"
    assert note_style_stage["summary"]["packs_completed"] == 1


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
    _write_response(
        response_path,
        {
            "sid": sid,
            "account_id": account_id,
            "answers": {"explanation": "Please help correct this."},
            "received_at": "2024-01-03T00:00:00Z",
        },
    )

    first = build_note_style_pack_for_account(sid, account_id, runs_root=runs_root)
    first_salt = first["prompt_salt"]

    _write_response(
        response_path,
        {
            "sid": sid,
            "account_id": account_id,
            "answers": {"explanation": "This is urgent and I dispute this account."},
            "received_at": "2024-01-03T00:10:00Z",
        },
    )

    updated = build_note_style_pack_for_account(sid, account_id, runs_root=runs_root)
    assert updated["status"] == "completed"
    assert updated["prompt_salt"] != first_salt

    paths = ensure_note_style_paths(runs_root, sid, create=False)
    account_paths = ensure_note_style_account_paths(paths, account_id, create=False)
    result_payload = json.loads(account_paths.result_file.read_text(encoding="utf-8"))
    assert result_payload["analysis"]["tone"]["value"] == "urgent"
    assert "dispute_resolution" in result_payload["analysis"]["emphasis"]["values"]


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

    result = build_note_style_pack_for_account(sid, account_id, runs_root=runs_root)
    assert result["status"] == "completed"

    sanitized = _sanitize_note_text(note)
    expected_hash = _normalized_hash(sanitized)
    expected_short_hash = expected_hash[:12]
    expected_salt = _expected_salt(sid, account_id, sanitized)

    paths = ensure_note_style_paths(runs_root, sid, create=False)
    account_paths = ensure_note_style_account_paths(paths, account_id, create=False)

    pack_payload = json.loads(account_paths.pack_file.read_text(encoding="utf-8"))
    result_payload = json.loads(account_paths.result_file.read_text(encoding="utf-8"))
    assert pack_payload["messages"][1]["content"] == sanitized
    assert pack_payload["note_hash"] == expected_short_hash
    assert result_payload["source_hash"] == expected_hash
    assert result_payload["note_hash"] == expected_short_hash
    assert result_payload["prompt_salt"] == expected_salt
    assert result["prompt_salt"] == expected_salt


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
    assert result["status"] == "skipped"
    assert result["reason"] == "empty_note"

    paths = ensure_note_style_paths(runs_root, sid, create=False)
    account_paths = ensure_note_style_account_paths(paths, account_id, create=False)

    assert not account_paths.pack_file.exists()
    assert not account_paths.result_file.exists()

    index_payload = json.loads(paths.index_file.read_text(encoding="utf-8"))
    assert index_payload["totals"] == {"total": 0, "completed": 0, "failed": 0}
    assert index_payload["items"] == []

    runflow_payload = json.loads((run_dir / "runflow.json").read_text(encoding="utf-8"))
    note_style_stage = runflow_payload["stages"]["note_style"]
    assert note_style_stage["status"] == "success"
    assert note_style_stage["summary"]["packs_completed"] == 0
