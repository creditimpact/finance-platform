from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from backend.core.services.ai_client import AIClient, AIConfig


class StubChatCompletions:
    def __init__(self, response):
        self.response = response
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return self.response


def _build_response(*, content: str | None, tool_arguments: object | None = None):
    message = SimpleNamespace(content=content)
    if tool_arguments is not None:
        message.tool_calls = [
            SimpleNamespace(function=SimpleNamespace(arguments=tool_arguments))
        ]
    else:
        message.tool_calls = None

    return SimpleNamespace(
        choices=[SimpleNamespace(message=message)],
        usage=SimpleNamespace(prompt_tokens=7, completion_tokens=3, total_tokens=10),
    )


@pytest.fixture
def client(monkeypatch):
    class DummyOpenAI:
        def __init__(self, **_: object):
            self.chat = SimpleNamespace(
                completions=StubChatCompletions(
                    SimpleNamespace(
                        choices=[
                            SimpleNamespace(
                                message=SimpleNamespace(content="{}", tool_calls=None)
                            )
                        ],
                        usage=None,
                    )
                )
            )

    monkeypatch.setattr("backend.core.services.ai_client.OpenAI", DummyOpenAI)

    config = AIConfig(api_key="test")
    return AIClient(config)


def test_chat_completion_content_mode(client):
    response = _build_response(content="{\"key\": \"value\"}")
    completions = StubChatCompletions(response)
    client._client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

    result = client.chat_completion(messages=[{"role": "user", "content": "hi"}])

    assert result["mode"] == "content"
    assert result["content_json"] == {"key": "value"}
    assert result["tool_json"] is None
    assert result["json"] == {"key": "value"}
    assert result["raw"] is response
    assert result["openai"] is response
    assert result["raw_content"] == "{\"key\": \"value\"}"
    assert result["raw_tool_arguments"] is None
    assert completions.kwargs["response_format"] == {"type": "json_object"}


def test_chat_completion_tool_mode(client):
    response = _build_response(content=None, tool_arguments="{\"tool\": true}")
    completions = StubChatCompletions(response)
    client._client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

    result = client.chat_completion(
        messages=[{"role": "user", "content": "hi"}],
        tools=[{"type": "function", "function": {"name": "noop", "parameters": {}}}],
    )

    assert result["mode"] == "tool"
    assert result["content_json"] is None
    assert result["tool_json"] == {"tool": True}
    assert result["json"] == {"tool": True}
    assert result["raw_content"] is None
    assert result["raw_tool_arguments"] == "{\"tool\": true}"
    assert "response_format" not in completions.kwargs


def test_chat_completion_tool_mode_with_dict_arguments(client):
    payload = {"tool": True, "nested": {"value": 3}}
    response = _build_response(content=None, tool_arguments=payload)
    completions = StubChatCompletions(response)
    client._client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

    result = client.chat_completion(
        messages=[{"role": "user", "content": "hi"}],
        tools=[{"type": "function", "function": {"name": "noop", "parameters": {}}}],
    )

    assert result["mode"] == "tool"
    assert result["content_json"] is None
    assert result["tool_json"] == payload
    assert result["json"] == payload
    expected_raw = json.dumps(payload, ensure_ascii=False)
    assert result["raw_tool_arguments"] == expected_raw
    assert result["raw_content"] is None

