from types import SimpleNamespace

from backend.core.services.ai_client import AIClient, AIConfig


def test_extra_headers_sanitized():
    client = AIClient(AIConfig(api_key="test"))

    captured = {}

    dummy_response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="{\"foo\": \"bar\"}", tool_calls=None)
            )
        ],
        usage=None,
    )

    class DummyCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return dummy_response

    client._client = SimpleNamespace(
        chat=SimpleNamespace(completions=DummyCompletions())
    )

    client.chat_completion(
        messages=[{"role": "user", "content": "hi"}],
        extra_headers={"X-Test": "héllo"},
    )

    assert captured["extra_headers"] == {"X-Test": "hllo"}
    assert captured["temperature"] == 0
    assert captured["top_p"] == 1
    assert captured["frequency_penalty"] == 0
    assert captured["presence_penalty"] == 0
