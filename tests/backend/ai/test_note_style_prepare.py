import json
from pathlib import Path
from typing import Any, Mapping

import pytest

import backend.ai.note_style as note_style_module
from backend.ai.note_style import (
    prepare_and_send,
    schedule_prepare_and_send,
    schedule_send_for_sid,
)
from backend.ai.note_style.tasks import note_style_send_account_task
from backend.ai.note_style_stage import build_note_style_pack_for_account
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


def test_prepare_and_send_builds_and_sends(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sid = "SID200"
    account_id = "idx-200"
    run_dir = tmp_path / sid
    response_dir = run_dir / "frontend" / "review" / "responses"

    account_dir = run_dir / "cases" / "accounts" / account_id
    account_dir.mkdir(parents=True, exist_ok=True)
    meta_payload = {"heading_guess": "Capital One"}
    bureaus_payload = {
        "transunion": {
            "account_type": "Credit Card",
            "account_status": "Open",
            "payment_status": "Current",
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

    manifest_payload = {
        "artifacts": {
            "cases": {
                "accounts": {
                    account_id: {
                        "dir": f"cases/accounts/{account_id}",
                        "meta": "meta.json",
                        "bureaus": "bureaus.json",
                        "tags": "tags.json",
                    }
                }
            }
        }
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    _write_response(
        response_dir / f"{account_id}.result.json",
        {
            "sid": sid,
            "account_id": account_id,
            "answers": {"explanation": "Please help, I already paid."},
        },
    )

    send_calls: list[dict[str, Any]] = []

    def _fake_send_apply_async(*, args, kwargs, task_id=None, expires=None):
        send_calls.append(
            {
                "args": args,
                "kwargs": kwargs,
                "task_id": task_id,
                "expires": expires,
            }
        )

    monkeypatch.setattr(
        "backend.ai.note_style.tasks.note_style_send_sid_task.apply_async",
        _fake_send_apply_async,
    )

    result = prepare_and_send(sid, runs_root=tmp_path)

    assert result["accounts_discovered"] == 1
    assert result["packs_built"] == 1
    assert result["processed_accounts"] == [account_id]
    assert len(send_calls) == 1
    send_call = send_calls[0]
    assert send_call["args"] == (sid,)
    assert send_call["kwargs"] == {"runs_root": str(tmp_path)}

    paths = ensure_note_style_paths(tmp_path, sid, create=False)
    account_paths = ensure_note_style_account_paths(paths, account_id, create=False)

    pack_lines = [
        line.strip()
        for line in account_paths.pack_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert len(pack_lines) == 1
    pack_payload = json.loads(pack_lines[0])
    assert pack_payload["messages"][1]["content"]["note_text"]
    assert not account_paths.result_file.exists()

    runflow_path = run_dir / "runflow.json"
    assert runflow_path.is_file()
    runflow_payload = json.loads(runflow_path.read_text(encoding="utf-8"))
    note_style_stage = runflow_payload["stages"]["note_style"]
    assert note_style_stage["status"] == "built"
    assert note_style_stage["results"]["completed"] == 0

    manifest_path = run_dir / "manifest.json"
    manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
    stage_status = manifest_data["ai"]["status"]["note_style"]
    assert stage_status["built"] is True
    assert stage_status["sent"] is False
    assert stage_status["completed_at"] is None


def test_note_style_send_account_task_processes_pack(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sid = "SID201A"
    account_id = "idx-201"
    run_dir = tmp_path / sid
    response_dir = run_dir / "frontend" / "review" / "responses"

    account_dir = run_dir / "cases" / "accounts" / account_id
    account_dir.mkdir(parents=True, exist_ok=True)
    meta_payload = {"heading_guess": "Capital One"}
    bureaus_payload = {
        "transunion": {
            "account_type": "Credit Card",
            "account_status": "Open",
            "payment_status": "Current",
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

    manifest_payload = {
        "artifacts": {
            "cases": {
                "accounts": {
                    account_id: {
                        "dir": f"cases/accounts/{account_id}",
                        "meta": "meta.json",
                        "bureaus": "bureaus.json",
                        "tags": "tags.json",
                    }
                }
            }
        }
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    _write_response(
        response_dir / f"{account_id}.result.json",
        {
            "sid": sid,
            "account_id": account_id,
            "answers": {"explanation": "Please help, I already paid."},
        },
    )

    build_note_style_pack_for_account(sid, account_id, runs_root=tmp_path)

    client = _StubClient()
    monkeypatch.setattr(
        "backend.ai.note_style_sender.get_ai_client", lambda: client
    )

    task_result = note_style_send_account_task(
        sid, account_id, runs_root=str(tmp_path)
    )

    assert task_result["processed"] is True
    assert client.calls

    paths = ensure_note_style_paths(tmp_path, sid, create=False)
    account_paths = ensure_note_style_account_paths(paths, account_id, create=False)
    assert account_paths.result_file.is_file()

    result_lines = [
        line.strip()
        for line in account_paths.result_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    stored_payload = json.loads(result_lines[0])
    assert stored_payload["analysis"]["tone"] == "Empathetic"

    manifest_path = run_dir / "manifest.json"
    manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
    stage_status = manifest_data["ai"]["status"]["note_style"]
    assert stage_status["built"] is True
    assert stage_status["sent"] is True
    assert isinstance(stage_status["completed_at"], str)

    runflow_payload = json.loads((run_dir / "runflow.json").read_text(encoding="utf-8"))
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


def test_schedule_prepare_and_send_invokes_celery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("NOTE_STYLE_DEBOUNCE_MS", raising=False)

    calls: list[dict[str, Any]] = []

    def _fake_apply_async(*, args, kwargs, countdown=None, task_id=None, expires=None):
        calls.append(
            {
                "args": args,
                "kwargs": kwargs,
                "countdown": countdown,
                "task_id": task_id,
                "expires": expires,
            }
        )

    monkeypatch.setattr(
        "backend.ai.note_style.tasks.note_style_prepare_and_send_task.apply_async",
        _fake_apply_async,
    )

    schedule_prepare_and_send("SID202", runs_root=tmp_path)

    assert len(calls) == 1
    call = calls[0]
    assert call["args"] == ("SID202",)
    assert call["kwargs"] == {"runs_root": str(tmp_path)}
    assert call["countdown"] == note_style_module._debounce_delay_seconds()
    if call["countdown"] and call["countdown"] > 0:
        assert call["task_id"] is not None


def test_schedule_prepare_and_send_dedupes_duplicate_tasks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("NOTE_STYLE_DEBOUNCE_MS", raising=False)

    class DuplicateTaskError(Exception):
        pass

    def _raise_duplicate(**kwargs: Any) -> None:  # type: ignore[override]
        raise DuplicateTaskError()

    monkeypatch.setattr(
        "backend.ai.note_style.tasks.note_style_prepare_and_send_task.apply_async",
        _raise_duplicate,
    )

    schedule_prepare_and_send("SID303", runs_root=tmp_path)


def test_schedule_send_for_sid_invokes_celery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(note_style_module.time, "time", lambda: 100.0)

    calls: list[dict[str, Any]] = []

    def _fake_apply_async(*, args, kwargs, task_id=None, expires=None):
        calls.append(
            {
                "args": args,
                "kwargs": kwargs,
                "task_id": task_id,
                "expires": expires,
            }
        )

    monkeypatch.setattr(
        "backend.ai.note_style.tasks.note_style_send_sid_task.apply_async",
        _fake_apply_async,
    )

    schedule_send_for_sid(
        "SID404", runs_root=tmp_path, trigger="unit", account_ids=("idx-1", "")
    )

    assert len(calls) == 1
    call = calls[0]
    assert call["args"] == ("SID404",)
    assert call["kwargs"] == {"runs_root": str(tmp_path)}
    assert call["task_id"] == "note-style.send:SID404:100"
    assert call["expires"] == 300


def test_schedule_send_for_sid_dedupes_duplicate_tasks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(note_style_module.time, "time", lambda: 42.0)

    class DuplicateTaskError(Exception):
        pass

    def _raise_duplicate(**kwargs: Any) -> None:  # type: ignore[override]
        raise DuplicateTaskError()

    monkeypatch.setattr(
        "backend.ai.note_style.tasks.note_style_send_sid_task.apply_async",
        _raise_duplicate,
    )

    schedule_send_for_sid("SID405")
