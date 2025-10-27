"""Tests for pipeline lifecycle hooks."""

from __future__ import annotations

from typing import Any

import pytest


class _TaskStub:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def apply_async(self, *, args: list[Any], queue: str) -> None:  # pragma: no cover - exercised in tests
        self.calls.append({"args": args, "queue": queue})


@pytest.fixture()
def task_stub(monkeypatch: pytest.MonkeyPatch) -> _TaskStub:
    from backend.api import tasks as tasks_module

    stub = _TaskStub()
    monkeypatch.setattr(tasks_module, "generate_frontend_packs_task", stub)
    return stub


def test_on_cases_built_default_enabled(monkeypatch: pytest.MonkeyPatch, task_stub: _TaskStub) -> None:
    """When the env flag is unset, the frontend task should be queued."""

    monkeypatch.delenv("FRONTEND_TRIGGER_AFTER_CASES", raising=False)

    from pipeline import hooks

    hooks.on_cases_built("SID-456")

    assert task_stub.calls == [{"args": ["SID-456"], "queue": "frontend"}]


def test_on_cases_built_respects_env_flag(monkeypatch: pytest.MonkeyPatch, task_stub: _TaskStub) -> None:
    """The frontend enqueue can be disabled through an environment flag."""

    monkeypatch.setenv("FRONTEND_TRIGGER_AFTER_CASES", "0")

    from pipeline import hooks

    hooks.on_cases_built("SID-789")

    assert task_stub.calls == []
