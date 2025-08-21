from backend.core.services.ai_client import AIClient, AIConfig


def test_extra_headers_sanitized():
    client = AIClient(AIConfig(api_key="test"))

    captured = {}

    class DummyCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return {"choices": []}

    class DummyChat:
        def __init__(self):
            self.completions = DummyCompletions()

    class DummyOpenAI:
        def __init__(self):
            self.chat = DummyChat()

    client._client = DummyOpenAI()

    client.chat_completion(
        messages=[{"role": "user", "content": "hi"}],
        extra_headers={"X-Test": "héllo"},
    )

    assert captured["extra_headers"] == {"X-Test": "hllo"}
