from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.ai.note_style_sender import send_note_style_packs_for_sid
from backend.ai.note_style_stage import build_note_style_pack_for_account
from backend.core.ai.paths import ensure_note_style_account_paths, ensure_note_style_paths


def _write_response(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


class _StubClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def chat_completion(self, *, model, messages, temperature):  # type: ignore[override]
        self.calls.append({"model": model, "messages": messages, "temperature": temperature})
        return {"choices": [{"message": {"content": "{}"}}]}


def test_note_style_sender_sends_built_pack(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    sid = "SID100"
    account_id = "idx-100"
    runs_root = tmp_path
    run_dir = runs_root / sid
    response_dir = run_dir / "frontend" / "review" / "responses"

    _write_response(
        response_dir / f"{account_id}.result.json",
        {
            "sid": sid,
            "account_id": account_id,
            "answers": {"explanation": "Please help, already paid."},
        },
    )

    build_note_style_pack_for_account(sid, account_id, runs_root=runs_root)

    client = _StubClient()
    monkeypatch.setattr("backend.ai.note_style_sender.get_ai_client", lambda: client)

    ingested: list[dict[str, object]] = []

    def _fake_ingest(**kwargs):
        ingested.append(kwargs)
        account_paths = kwargs["account_paths"]
        result_path = account_paths.result_file
        result_payload = {
            "sid": kwargs["sid"],
            "account_id": kwargs["account_id"],
            "response": {"content": "ok"},
        }
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps(result_payload), encoding="utf-8")
        return result_path

    monkeypatch.setattr("backend.ai.note_style_sender.ingest_note_style_result", _fake_ingest)

    caplog.set_level("INFO", logger="backend.ai.note_style_sender")

    processed = send_note_style_packs_for_sid(sid, runs_root=runs_root)
    assert processed == [account_id]
    assert len(client.calls) == 1
    assert len(ingested) == 1

    paths = ensure_note_style_paths(runs_root, sid, create=False)
    account_paths = ensure_note_style_account_paths(paths, account_id, create=False)

    index_payload = json.loads(paths.index_file.read_text(encoding="utf-8"))
    packs = index_payload["packs"]
    assert packs[0]["status"] == "completed"
    assert "sent_at" in packs[0]
    assert packs[0]["result"] == account_paths.result_file.relative_to(paths.base).as_posix()

    messages = [record.message for record in caplog.records if "STYLE_SEND" in record.message]
    assert any("STYLE_SEND_ACCOUNT_START" in message for message in messages)
    assert any("STYLE_SEND_MODEL_CALL" in message for message in messages)
    assert any("STYLE_SEND_RESULTS_WRITTEN" in message for message in messages)
    assert any("STYLE_SEND_ACCOUNT_END" in message for message in messages)


def test_note_style_sender_skips_completed_entries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sid = "SID101"
    account_id = "idx-101"
    runs_root = tmp_path
    run_dir = runs_root / sid
    response_dir = run_dir / "frontend" / "review" / "responses"

    _write_response(
        response_dir / f"{account_id}.result.json",
        {
            "sid": sid,
            "account_id": account_id,
            "answers": {"explanation": "Please fix"},
        },
    )

    build_note_style_pack_for_account(sid, account_id, runs_root=runs_root)

    client = _StubClient()
    monkeypatch.setattr("backend.ai.note_style_sender.get_ai_client", lambda: client)

    def _fake_ingest(**kwargs):
        return kwargs["account_paths"].result_file

    monkeypatch.setattr("backend.ai.note_style_sender.ingest_note_style_result", _fake_ingest)

    processed_first = send_note_style_packs_for_sid(sid, runs_root=runs_root)
    assert processed_first == [account_id]
    assert len(client.calls) == 1

    processed_second = send_note_style_packs_for_sid(sid, runs_root=runs_root)
    assert processed_second == []
    assert len(client.calls) == 1


def test_note_style_sender_raises_when_pack_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sid = "SID102"
    account_id = "idx-102"
    runs_root = tmp_path
    run_dir = runs_root / sid
    response_dir = run_dir / "frontend" / "review" / "responses"

    _write_response(
        response_dir / f"{account_id}.result.json",
        {
            "sid": sid,
            "account_id": account_id,
            "answers": {"explanation": "Need support"},
        },
    )

    build_note_style_pack_for_account(sid, account_id, runs_root=runs_root)

    paths = ensure_note_style_paths(runs_root, sid, create=False)
    account_paths = ensure_note_style_account_paths(paths, account_id, create=False)
    account_paths.pack_file.unlink()

    client = _StubClient()
    monkeypatch.setattr("backend.ai.note_style_sender.get_ai_client", lambda: client)
    monkeypatch.setattr(
        "backend.ai.note_style_sender.ingest_note_style_result",
        lambda **_: account_paths.result_file,
    )

    with pytest.raises(FileNotFoundError):
        send_note_style_packs_for_sid(sid, runs_root=runs_root)

    index_payload = json.loads(paths.index_file.read_text(encoding="utf-8"))
    packs = index_payload["packs"]
    assert packs[0]["status"] == "built"
    assert "sent_at" not in packs[0]
