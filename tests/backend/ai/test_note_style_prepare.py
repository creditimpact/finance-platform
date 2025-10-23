import json
from pathlib import Path
from typing import Any, Mapping

import pytest

import backend.ai.note_style as note_style_module
from backend.ai.note_style import prepare_and_send, schedule_prepare_and_send
from backend.core.ai.paths import ensure_note_style_account_paths, ensure_note_style_paths


def _write_response(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


class _StubClient:
    def __init__(self, *, response: Mapping[str, Any] | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self._response_payload = response or {
            "tone": "Empathetic",
            "context_hints": {
                "timeframe": {"month": "April", "relative": "Last month"},
                "topic": "Payment_Dispute",
                "entities": {"creditor": "capital one", "amount": "$123.45 USD"},
            },
            "emphasis": ["paid_already", "support_request"],
            "confidence": 0.92,
            "risk_flags": ["follow_up"],
        }

    def chat_completion(self, *, model, messages, temperature):  # type: ignore[override]
        self.calls.append({"model": model, "messages": messages, "temperature": temperature})
        return {
            "choices": [
                {"message": {"content": json.dumps(self._response_payload)}}
            ]
        }


def test_prepare_and_send_builds_and_sends(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sid = "SID200"
    account_id = "idx-200"
    run_dir = tmp_path / sid
    response_dir = run_dir / "frontend" / "review" / "responses"

    _write_response(
        response_dir / f"{account_id}.result.json",
        {
            "sid": sid,
            "account_id": account_id,
            "answers": {"explanation": "Please help, I already paid."},
        },
    )

    client = _StubClient()
    monkeypatch.setattr(
        "backend.ai.note_style_sender.get_ai_client", lambda: client
    )

    result = prepare_and_send(sid, runs_root=tmp_path)

    assert result["accounts_discovered"] == 1
    assert result["packs_built"] == 1
    assert result["processed_accounts"] == [account_id]
    assert client.calls

    paths = ensure_note_style_paths(tmp_path, sid, create=False)
    account_paths = ensure_note_style_account_paths(paths, account_id, create=False)

    pack_lines = [
        line.strip()
        for line in account_paths.pack_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    result_lines = [
        line.strip()
        for line in account_paths.result_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert len(pack_lines) == 1
    assert len(result_lines) == 1

    stored_payload = json.loads(result_lines[0])
    assert stored_payload["sid"] == sid
    assert stored_payload["account_id"] == account_id
    assert stored_payload["analysis"]["tone"] == "empathetic"

    runflow_path = run_dir / "runflow.json"
    assert runflow_path.is_file()
    runflow_payload = json.loads(runflow_path.read_text(encoding="utf-8"))
    note_style_stage = runflow_payload["stages"]["note_style"]
    assert note_style_stage["status"] == "success"
    assert note_style_stage["results"]["completed"] >= 1


def test_prepare_and_send_records_empty_stage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sid = "SID201"
    run_dir = tmp_path / sid
    run_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(
        note_style_module, "runflow_barriers_refresh", lambda _sid: None
    )
    monkeypatch.setattr(
        note_style_module, "reconcile_umbrella_barriers", lambda _sid, runs_root=None: {}
    )

    result = prepare_and_send(sid, runs_root=tmp_path)

    assert result["accounts_discovered"] == 0
    assert result["processed_accounts"] == []

    runflow_path = run_dir / "runflow.json"
    assert runflow_path.is_file()
    payload = json.loads(runflow_path.read_text(encoding="utf-8"))
    stage = payload["stages"]["note_style"]
    assert stage["status"] == "success"
    assert stage["empty_ok"] is True


def test_schedule_prepare_and_send_invokes_prepare(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    note_style_module._PENDING.clear()

    calls: list[tuple[str, Any]] = []

    def _fake_prepare(sid: str, *, runs_root: Path | str | None = None) -> None:
        calls.append((sid, runs_root))

    monkeypatch.setattr(note_style_module, "prepare_and_send", _fake_prepare)

    try:
        schedule_prepare_and_send("SID202", runs_root=tmp_path)
    finally:
        note_style_module._PENDING.clear()

    assert calls == [("SID202", tmp_path)]
