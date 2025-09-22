import json
from typing import Any

import pytest

from backend.core.ai import adjudicator


class _FakeResponse:
    def __init__(self, payload: dict[str, Any] | None = None) -> None:
        if payload is None:
            payload = {"decision": "merge", "reason": "test"}
        self._payload = payload

    def raise_for_status(self) -> None:  # pragma: no cover - simple stub
        return None

    def json(self) -> dict[str, Any]:
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(self._payload, ensure_ascii=False)
                    }
                }
            ]
        }


def _clear_ai_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in ("OPENAI_API_KEY", "AI_MODEL", "OPENAI_PROJECT_ID"):
        monkeypatch.delenv(key, raising=False)


def test_decide_merge_requires_project_id_for_project_key(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_ai_env(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-proj-missing")
    monkeypatch.setenv("AI_MODEL", "unit-test-model")

    called: dict[str, Any] = {}

    def fake_post(*args: Any, **kwargs: Any) -> _FakeResponse:
        called["args"] = args
        return _FakeResponse()

    monkeypatch.setattr(adjudicator.httpx, "post", fake_post)

    with pytest.raises(RuntimeError, match="OPENAI_PROJECT_ID must be set"):
        adjudicator.decide_merge_or_different({}, timeout=5)

    assert called == {}


def test_decide_merge_attaches_project_header(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_ai_env(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-proj-abc123")
    monkeypatch.setenv("OPENAI_PROJECT_ID", "proj-789")
    monkeypatch.setenv("AI_MODEL", "merge-model")

    recorded: dict[str, Any] = {}

    def fake_post(url: str, *, headers: dict[str, str], json: dict[str, Any], timeout: int) -> _FakeResponse:
        recorded["url"] = url
        recorded["headers"] = dict(headers)
        recorded["json"] = json
        recorded["timeout"] = timeout
        return _FakeResponse()

    monkeypatch.setattr(adjudicator.httpx, "post", fake_post)

    result = adjudicator.decide_merge_or_different({}, timeout=7)

    assert result == {"decision": "merge", "reason": "test"}
    assert recorded["headers"]["Authorization"].startswith("Bearer sk-proj-abc123")
    assert recorded["headers"]["OpenAI-Project"] == "proj-789"
