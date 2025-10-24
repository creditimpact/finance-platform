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
                " ",
            ],
        }

    def chat_completion(self, *, model, messages, temperature):  # type: ignore[override]
        self.calls.append({"model": model, "messages": messages, "temperature": temperature})
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
    assert stored_payload["sid"] == sid
    assert stored_payload["account_id"] == account_id
    assert isinstance(stored_payload["evaluated_at"], str)

    pack_payload = json.loads(
        account_paths.pack_file.read_text(encoding="utf-8").splitlines()[0]
    )
    assert stored_payload["prompt_salt"] == pack_payload["prompt_salt"]
    assert "fingerprint" not in stored_payload
    assert stored_payload["fingerprint_hash"] == pack_payload["fingerprint_hash"]
    analysis = stored_payload["analysis"]
    assert analysis["tone"] == "empathetic"
    assert analysis["emphasis"] == ["paid_already", "support_request"]
    context = analysis["context_hints"]
    assert context["topic"] == "payment_dispute"
    timeframe = context["timeframe"]
    assert timeframe.get("relative") in {"Last month", "last_month"}
    month_value = timeframe.get("month")
    if month_value is not None:
        assert str(month_value).lower().startswith("apr")
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

    index_payload = json.loads(paths.index_file.read_text(encoding="utf-8"))
    packs = index_payload["packs"]
    assert packs[0]["status"] == "completed"
    assert "sent_at" not in packs[0]
    assert (
        packs[0]["result_path"]
        == account_paths.result_file.relative_to(paths.base).as_posix()
    )
    assert packs[0]["completed_at"] == stored_payload["evaluated_at"]

    messages = [record.message for record in caplog.records if "STYLE_SEND" in record.message]
    assert any("STYLE_SEND_ACCOUNT_START" in message for message in messages)
    assert any("STYLE_SEND_MODEL_CALL" in message for message in messages)
    assert any("STYLE_SEND_RESULTS_WRITTEN" in message for message in messages)
    assert any("STYLE_SEND_ACCOUNT_END" in message for message in messages)

    structured_records = [
        json.loads(record.getMessage())
        for record in caplog.records
        if record.name == "backend.ai.note_style_sender"
        and record.getMessage().startswith("{")
    ]
    assert any(
        entry.get("event") == "NOTE_STYLE_SENT_OK" and entry.get("account_id") == account_id
        for entry in structured_records
    )


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
    with pytest.raises(FileNotFoundError):
        send_note_style_packs_for_sid(sid, runs_root=runs_root)

    index_payload = json.loads(paths.index_file.read_text(encoding="utf-8"))
    packs = index_payload["packs"]
    assert packs[0]["status"] == "built"
