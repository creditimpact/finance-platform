from __future__ import annotations

from typing import Any

import pytest

from backend.ai.note_style import tasks


class DummyRedis:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.deleted: list[str] = []

    def set(self, key: str, value: Any, *, nx: bool | None = None, ex: int | None = None) -> bool:
        self.calls.append(key)
        # First acquisition succeeds; subsequent attempts simulate contention.
        return len(self.calls) == 1

    def delete(self, key: str) -> int:
        self.deleted.append(key)
        return 1


@pytest.fixture
def dummy_redis(monkeypatch: pytest.MonkeyPatch) -> DummyRedis:
    redis = DummyRedis()
    monkeypatch.setattr(tasks, "redis", redis)
    return redis


def test_prepare_and_send_skips_when_locked(monkeypatch: pytest.MonkeyPatch, dummy_redis: DummyRedis) -> None:
    sid = "SID123"

    monkeypatch.setattr(tasks.config, "NOTE_STYLE_AUTOSEND", True)
    monkeypatch.setattr(tasks, "_note_style_has_packs", lambda *_, **__: True)
    monkeypatch.setattr(tasks.runs, "get_stage_status", lambda *_, **__: None)

    prepared_result = {"sid": sid, "built": 1, "sent": 1}
    monkeypatch.setattr(tasks, "prepare_and_send", lambda *_, **__: prepared_result)

    result_first = tasks.note_style_prepare_and_send_task(sid, runs_root="/tmp")
    assert result_first == prepared_result
    assert dummy_redis.calls == ["note-style:prepare:SID123"]
    assert dummy_redis.deleted == ["note-style:prepare:SID123"]

    result_second = tasks.note_style_prepare_and_send_task(sid, runs_root="/tmp")
    assert result_second["skipped"] == "locked"
    assert dummy_redis.calls == ["note-style:prepare:SID123", "note-style:prepare:SID123"]
