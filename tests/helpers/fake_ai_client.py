from types import SimpleNamespace
from typing import Any, Dict, List


class FakeAIClient:
    """Simple stand-in for :class:`services.ai_client.AIClient` in tests."""

    def __init__(self):
        self.chat_payloads: List[Dict[str, Any]] = []
        self.responses: List[str] = []
        self.chat_responses: List[str] = []

    def add_chat_response(self, content: str) -> None:
        self.chat_responses.append(content)

    def add_response(self, content: str) -> None:
        self.responses.append(content)

    def chat_completion(self, *, messages, **kwargs):
        self.chat_payloads.append({"messages": messages, **kwargs})
        content = self.chat_responses.pop(0) if self.chat_responses else ""
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )

    def response_json(self, *, prompt, **kwargs):
        self.chat_payloads.append({"prompt": prompt, **kwargs})
        content = self.responses.pop(0) if self.responses else ""
        return SimpleNamespace(
            output=[SimpleNamespace(content=[SimpleNamespace(text=content)])]
        )
