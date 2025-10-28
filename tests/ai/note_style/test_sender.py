from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from backend.ai.note_style import tasks as note_style_tasks
from backend.ai.note_style_sender import send_note_style_packs_for_sid

from ._helpers import prime_stage


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


def _result_payload(account_id: str) -> dict[str, Any]:
    return {
        "status": "completed",
        "account_id": account_id,
        "analysis": {"note": f"analysis for {account_id}"},
    }


def test_sender_skips_accounts_with_results(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sid = "SID-SENDER"
    accounts = ["idx-601", "idx-602", "idx-603"]

    prime_stage(
        tmp_path,
        sid,
        expected_accounts=accounts,
        built_accounts=accounts,
    )

    send_calls: list[str] = []

    def _fake_send_pack_payload(*, account_id: str, account_paths, **kwargs: Any) -> bool:
        send_calls.append(account_id)
        account_paths.result_file.write_text(
            json.dumps(_result_payload(account_id)),
            encoding="utf-8",
        )
        return True

    monkeypatch.setattr(
        "backend.ai.note_style_sender._send_pack_payload",
        _fake_send_pack_payload,
    )
    monkeypatch.setattr(
        "backend.ai.note_style_sender.get_ai_client",
        lambda: object(),
    )

    processed_first = send_note_style_packs_for_sid(sid, runs_root=tmp_path)
    assert set(processed_first) == set(accounts)
    assert send_calls == accounts

    processed_second = send_note_style_packs_for_sid(sid, runs_root=tmp_path)
    assert processed_second == []
    assert send_calls == accounts

