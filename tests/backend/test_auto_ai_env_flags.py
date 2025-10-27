"""Tests ensuring auto AI tasks respect environment gating."""

from __future__ import annotations

from typing import Any

import pytest


@pytest.mark.parametrize(
    "env_value,expected_calls",
    [("0", []), ("1", ["SID-123"]), (None, ["SID-123"])],
)
def test_maybe_autobuild_review_honors_generate_frontend_env(monkeypatch, env_value, expected_calls):
    """_maybe_autobuild_review should defer to GENERATE_FRONTEND_ON_VALIDATION."""

    from backend.pipeline import auto_ai_tasks

    captured: list[Any] = []

    class _TaskStub:
        def delay(self, sid: str) -> None:
            captured.append(sid)

    monkeypatch.setattr(auto_ai_tasks, "generate_frontend_packs_task", _TaskStub())

    if env_value is None:
        monkeypatch.delenv("GENERATE_FRONTEND_ON_VALIDATION", raising=False)
    else:
        monkeypatch.setenv("GENERATE_FRONTEND_ON_VALIDATION", env_value)

    auto_ai_tasks._maybe_autobuild_review("SID-123")

    assert captured == expected_calls
