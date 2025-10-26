from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pytest

from backend.ai.note_style_sender import send_note_style_packs_for_sid
from backend.ai.note_style_stage import build_note_style_pack_for_account
from backend.core.ai.paths import ensure_note_style_account_paths, ensure_note_style_paths


def _write_response(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_manifest(run_dir: Path, account_id: str) -> Path:
    account_dir = run_dir / "cases" / "accounts" / account_id
    account_dir.mkdir(parents=True, exist_ok=True)
    manifest_payload = {
        "artifacts": {
            "cases": {
                "accounts": {
                    account_id: {
                        "dir": "cases/accounts/" + account_id,
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


class _StubClient:
    def __init__(self, *, response: Mapping[str, Any] | None = None) -> None:
        self.calls: list[dict[str, object]] = []
        self._response_payload = response or {
            "tone": "Empathetic",
            "context_hints": {
                "timeframe": {"month": "April", "relative": "Last month"},
                "topic": "Payment_Dispute",
                "entities": {"creditor": "capital one", "amount": "$123.45 USD"},
            },
            "emphasis": ["paid_already", "Custom", "support_request"],
            "confidence": 0.91,
            "risk_flags": [
                "Follow_Up",
                "duplicate",
                "FOLLOW_UP",
                "Mixed Language",
                "ALL CAPS",
                "possible-template copy",
            ],
        }

    def chat_completion(self, *, model, messages, temperature, **kwargs):  # type: ignore[override]
        self.calls.append(
            {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "kwargs": kwargs,
            }
        )
        return {
            "choices": [
                {"message": {"content": json.dumps(self._response_payload)}}
            ]
        }


def test_note_style_sender_sends_built_pack(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    sid = "SID100"
    account_id = "idx-100"
    runs_root = tmp_path
    run_dir = runs_root / sid
    response_dir = run_dir / "frontend" / "review" / "responses"

    _write_manifest(run_dir, account_id)

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

    caplog.set_level("INFO", logger="backend.ai.note_style_sender")

    processed = send_note_style_packs_for_sid(sid, runs_root=runs_root)
    assert processed == [account_id]
    assert len(client.calls) == 1

    paths = ensure_note_style_paths(runs_root, sid, create=False)
    account_paths = ensure_note_style_account_paths(paths, account_id, create=False)

    result_lines = [
        line
        for line in account_paths.result_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(result_lines) == 1
    stored_payload = json.loads(result_lines[0])
    assert set(stored_payload.keys()) == {
        "sid",
        "account_id",
        "evaluated_at",
        "analysis",
        "note_metrics",
    }
    assert stored_payload["sid"] == sid
    assert stored_payload["account_id"] == account_id
    assert stored_payload["evaluated_at"].endswith("Z")

    pack_payload = json.loads(
        account_paths.pack_file.read_text(encoding="utf-8").splitlines()[0]
    )
    note_text = pack_payload["note_text"]
    assert stored_payload["note_metrics"] == {
        "char_len": len(note_text),
        "word_len": len(note_text.split()),
    }
    assert "prompt_salt" not in stored_payload
    assert "prompt_salt" not in pack_payload
    assert "fingerprint_hash" not in stored_payload
    assert "fingerprint_hash" not in pack_payload
    assert "fingerprint" not in stored_payload
    analysis = stored_payload["analysis"]
    assert analysis["tone"] == "Empathetic"
    assert analysis["emphasis"] == ["paid_already", "custom", "support_request"]
    context = analysis["context_hints"]
    assert context["topic"] == "payment_dispute"
    timeframe = context["timeframe"]
    assert timeframe.get("relative") == "last_month"
    assert timeframe.get("month") in {None, "2024-04-01"}
    entities = context["entities"]
    assert entities["creditor"] == "capital one"
    assert entities["amount"] == pytest.approx(123.45)
    assert analysis["risk_flags"] == [
        "follow_up",
        "duplicate",
        "mixed_language",
        "all_caps",
        "possible_template_copy",
    ]

    note_metrics = stored_payload.get("note_metrics")
    assert isinstance(note_metrics, Mapping)
    assert note_metrics.get("char_len") > 0
    assert note_metrics.get("word_len") > 0
    assert set(note_metrics.keys()) == {"char_len", "word_len"}

    assert "account_context" not in stored_payload
    assert "bureaus_summary" not in stored_payload
    assert pack_payload["messages"][1]["content"]["note_text"]

    index_payload = json.loads(paths.index_file.read_text(encoding="utf-8"))
    packs = index_payload["packs"]
    assert packs[0]["status"] == "completed"
    assert "sent_at" not in packs[0]
    assert (
        packs[0]["result_path"]
        == account_paths.result_file.relative_to(paths.base).as_posix()
    )
    assert isinstance(packs[0].get("completed_at"), str)

    messages = [
        record.getMessage()
        for record in caplog.records
        if record.name == "backend.ai.note_style_sender"
    ]
    assert any("STYLE_SEND_ACCOUNT_START" in message for message in messages)
    assert any("STYLE_SEND_MODEL_CALL" in message for message in messages)
    assert any("STYLE_SEND_ACCOUNT_END" in message for message in messages)
    assert any("NOTE_STYLE_SENT" in message for message in messages)

    structured_records = [
        json.loads(record.getMessage())
        for record in caplog.records
        if record.name == "backend.ai.note_style_sender"
        and record.getMessage().startswith("{")
    ]
    assert any(
        entry.get("event") == "NOTE_STYLE_SENT" and entry.get("account_id") == account_id
        for entry in structured_records
    )

    call_kwargs = client.calls[0]["kwargs"]
    assert call_kwargs.get("response_format") == "json_object"


def test_note_style_sender_skips_completed_entries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sid = "SID101"
    account_id = "idx-101"
    runs_root = tmp_path
    run_dir = runs_root / sid
    response_dir = run_dir / "frontend" / "review" / "responses"

    _write_manifest(run_dir, account_id)

    _write_response(
        response_dir / f"{account_id}.result.json",
        {
            "sid": sid,
            "account_id": account_id,
            "answers": {
                "explanation": "Please fix the errors on this account."
            },
        },
    )

    build_note_style_pack_for_account(sid, account_id, runs_root=runs_root)

    client = _StubClient()
    monkeypatch.setattr("backend.ai.note_style_sender.get_ai_client", lambda: client)

    processed_first = send_note_style_packs_for_sid(sid, runs_root=runs_root)
    assert processed_first == [account_id]
    assert len(client.calls) == 1

    processed_second = send_note_style_packs_for_sid(sid, runs_root=runs_root)
    assert processed_second == [account_id]
    assert len(client.calls) == 2


def test_note_style_sender_skips_when_existing_result_matches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sid = "SID200"
    account_id = "idx-200"
    runs_root = tmp_path
    run_dir = runs_root / sid
    response_dir = run_dir / "frontend" / "review" / "responses"

    _write_manifest(run_dir, account_id)

    _write_response(
        response_dir / f"{account_id}.result.json",
        {
            "sid": sid,
            "account_id": account_id,
            "answers": {"explanation": "Already fixed, thanks."},
        },
    )

    build_note_style_pack_for_account(sid, account_id, runs_root=runs_root)

    paths = ensure_note_style_paths(runs_root, sid, create=False)
    account_paths = ensure_note_style_account_paths(paths, account_id, create=False)

    pack_payload = json.loads(
        account_paths.pack_file.read_text(encoding="utf-8").splitlines()[0]
    )
    note_text = pack_payload["note_text"]
    final_result = {
        "sid": sid,
        "account_id": account_id,
        "note_metrics": {
            "char_len": len(note_text),
            "word_len": len(note_text.split()),
        },
        "analysis": {
            "tone": "neutral",
            "context_hints": {
                "timeframe": {"month": None, "relative": None},
                "topic": "other",
                "entities": {"creditor": None, "amount": None},
            },
            "emphasis": [],
            "confidence": 0.7,
            "risk_flags": ["follow_up"],
        },
    }
    account_paths.result_file.write_text(
        json.dumps(final_result, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    client = _StubClient()
    monkeypatch.setattr("backend.ai.note_style_sender.get_ai_client", lambda: client)

    processed = send_note_style_packs_for_sid(sid, runs_root=runs_root)

    assert processed == [account_id]
    assert len(client.calls) == 1

    updated_index = json.loads(paths.index_file.read_text(encoding="utf-8"))
    entry = updated_index["packs"][0]
    assert entry["status"] == "completed"
    assert "note_hash" not in entry
    assert entry.get("result_path") == account_paths.result_file.relative_to(paths.base).as_posix()

def test_note_style_sender_raises_when_pack_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sid = "SID102"
    account_id = "idx-102"
    runs_root = tmp_path
    run_dir = runs_root / sid
    response_dir = run_dir / "frontend" / "review" / "responses"

    _write_manifest(run_dir, account_id)

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
    processed = send_note_style_packs_for_sid(sid, runs_root=runs_root)
    assert processed == []
    assert len(client.calls) == 0


def test_note_style_sender_strips_debug_message_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sid = "SID300"
    account_id = "idx-300"
    runs_root = tmp_path
    run_dir = runs_root / sid
    response_dir = run_dir / "frontend" / "review" / "responses"

    _write_manifest(run_dir, account_id)

    _write_response(
        response_dir / f"{account_id}.result.json",
        {
            "sid": sid,
            "account_id": account_id,
            "answers": {"explanation": "Need info."},
        },
    )

    build_note_style_pack_for_account(sid, account_id, runs_root=runs_root)

    paths = ensure_note_style_paths(runs_root, sid, create=False)
    account_paths = ensure_note_style_account_paths(paths, account_id, create=False)

    payload = json.loads(account_paths.pack_file.read_text(encoding="utf-8").splitlines()[0])
    payload["messages"][1]["debug_snapshot"] = {"should": "not-travel"}
    payload["messages"][1]["raw_payload"] = {"secret": True}
    account_paths.pack_file.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")

    client = _StubClient()
    monkeypatch.setattr("backend.ai.note_style_sender.get_ai_client", lambda: client)

    processed = send_note_style_packs_for_sid(sid, runs_root=runs_root)

    assert processed == [account_id]
    assert len(client.calls) == 1

    call_messages = client.calls[0]["messages"]
    assert isinstance(call_messages, list)
    user_entry = call_messages[1]
    assert "debug_snapshot" not in user_entry
    assert "raw_payload" not in user_entry


def test_note_style_sender_ignores_debug_snapshot_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sid = "SID301"
    account_id = "idx-301"
    runs_root = tmp_path
    run_dir = runs_root / sid
    response_dir = run_dir / "frontend" / "review" / "responses"

    _write_manifest(run_dir, account_id)

    _write_response(
        response_dir / f"{account_id}.result.json",
        {
            "sid": sid,
            "account_id": account_id,
            "answers": {"explanation": "Need info."},
        },
    )

    build_note_style_pack_for_account(sid, account_id, runs_root=runs_root)

    paths = ensure_note_style_paths(runs_root, sid, create=False)
    debug_candidate = paths.debug_dir / "acc_debug.jsonl"
    debug_candidate.parent.mkdir(parents=True, exist_ok=True)
    debug_candidate.write_text(
        json.dumps({
            "messages": [
                {"role": "system", "content": "bad"},
                {"role": "user", "content": "bad"},
            ]
        }, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )

    client = _StubClient()
    monkeypatch.setattr("backend.ai.note_style_sender.get_ai_client", lambda: client)
    monkeypatch.setenv("NOTE_STYLE_PACK_GLOB", "**/*.jsonl")

    processed = send_note_style_packs_for_sid(sid, runs_root=runs_root)

    assert processed == [account_id]
    assert len(client.calls) == 1
