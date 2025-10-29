from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from backend.ai.note_style import tasks as note_style_tasks
from backend.ai.note_style.tasks import note_style_prepare_and_send_task
from backend.runflow.umbrella import schedule_note_style_after_validation

from ._helpers import prime_stage


class _TaskStub:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def delay(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover - simple recorder
        self.calls.append((args, kwargs))


class _RedisStub:
    def __init__(self) -> None:
        self.set_calls: list[str] = []
        self.deleted: list[str] = []

    def set(self, key: str, value: Any, *, nx: bool | None = None, ex: int | None = None) -> bool:
        self.set_calls.append(key)
        return True

    def delete(self, key: str) -> int:
        self.deleted.append(key)
        return 1


@pytest.fixture(autouse=True)
def _patch_redis(monkeypatch: pytest.MonkeyPatch) -> _RedisStub:
    redis = _RedisStub()
    monkeypatch.setattr(note_style_tasks, "redis", redis)
    return redis


def _write_manifest(path: Path, sid: str, *, validation_status: str, note_style_status: dict[str, Any] | None = None) -> None:
    payload = {
        "sid": sid,
        "ai": {
            "status": {
                "validation": {
                    "status": validation_status,
                    "completed_at": "2024-01-01T00:00:00Z",
                },
                "note_style": note_style_status or {"status": "pending", "sent": False},
            }
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def test_autosend_enqueues_when_stage_built(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sid = "SID-AUTO-BUILT"
    accounts = ["idx-301", "idx-302", "idx-303"]
    run_dir = tmp_path / sid

    prime_stage(
        tmp_path,
        sid,
        expected_accounts=accounts,
        built_accounts=accounts,
    )

    _write_manifest(run_dir / "manifest.json", sid, validation_status="success")

    task_stub = _TaskStub()
    monkeypatch.setattr(
        "backend.ai.note_style.tasks.note_style_prepare_and_send_task",
        task_stub,
    )

    decisions: list[dict[str, Any]] = []

    def _record_decision(**kwargs: Any) -> None:
        decisions.append(kwargs)

    monkeypatch.setattr(
        "backend.runflow.umbrella._log_autosend_decision",
        _record_decision,
    )

    schedule_note_style_after_validation(sid, run_dir=run_dir)

    assert task_stub.calls == [((sid,), {"runs_root": str(tmp_path)})]
    assert any(entry.get("reason") == "enqueued" for entry in decisions)


def test_autosend_skips_for_empty_stage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sid = "SID-AUTO-EMPTY"
    run_dir = tmp_path / sid
    run_dir.mkdir(parents=True, exist_ok=True)

    _write_manifest(run_dir / "manifest.json", sid, validation_status="success")

    task_stub = _TaskStub()
    monkeypatch.setattr(
        "backend.ai.note_style.tasks.note_style_prepare_and_send_task",
        task_stub,
    )

    schedule_note_style_after_validation(sid, run_dir=run_dir)

    assert task_stub.calls == []


def test_autosend_skips_when_stage_terminal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sid = "SID-AUTO-DONE"
    accounts = ["idx-401"]
    run_dir = tmp_path / sid

    prime_stage(
        tmp_path,
        sid,
        expected_accounts=accounts,
        built_accounts=accounts,
        completed_accounts=accounts,
    )

    _write_manifest(
        run_dir / "manifest.json",
        sid,
        validation_status="success",
        note_style_status={"status": "success", "sent": True},
    )

    task_stub = _TaskStub()
    monkeypatch.setattr(
        "backend.ai.note_style.tasks.note_style_prepare_and_send_task",
        task_stub,
    )

    schedule_note_style_after_validation(sid, run_dir=run_dir)

    assert task_stub.calls == []


def test_autosend_skips_when_stage_failed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sid = "SID-AUTO-FAILED"
    accounts = ["idx-601", "idx-602"]
    run_dir = tmp_path / sid

    prime_stage(
        tmp_path,
        sid,
        expected_accounts=accounts,
        built_accounts=accounts,
        failed_accounts=accounts,
    )

    _write_manifest(run_dir / "manifest.json", sid, validation_status="success")

    task_stub = _TaskStub()
    monkeypatch.setattr(
        "backend.ai.note_style.tasks.note_style_prepare_and_send_task",
        task_stub,
    )

    decisions: list[dict[str, Any]] = []

    def _record_decision(**kwargs: Any) -> None:
        decisions.append(kwargs)

    monkeypatch.setattr(
        "backend.runflow.umbrella._log_autosend_decision",
        _record_decision,
    )

    schedule_note_style_after_validation(sid, run_dir=run_dir)

    assert task_stub.calls == []
    assert any(entry.get("reason") == "already_complete" for entry in decisions)


def test_prepare_task_ignores_corrupt_terminal_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sid = "SID-CORRUPT"
    accounts = ["idx-501", "idx-502"]

    prime_stage(
        tmp_path,
        sid,
        expected_accounts=accounts,
        built_accounts=accounts,
    )

    run_dir = tmp_path / sid
    run_dir.mkdir(parents=True, exist_ok=True)
    runflow_path = run_dir / "runflow.json"
    runflow_payload = {"stages": {"note_style": {"status": "success"}}}
    runflow_path.write_text(json.dumps(runflow_payload, ensure_ascii=False), encoding="utf-8")

    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def _prepare_and_send_stub(*args: Any, **kwargs: Any) -> dict[str, Any]:
        calls.append((args, kwargs))
        return {"sid": sid, "built": len(accounts), "sent": len(accounts)}

    monkeypatch.setattr(note_style_tasks, "prepare_and_send", _prepare_and_send_stub)

    result = note_style_prepare_and_send_task(sid, runs_root=tmp_path)

    assert result["built"] == len(accounts)
    assert calls == [((sid,), {"runs_root": tmp_path})]

