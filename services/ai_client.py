from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Dict

from openai import OpenAI


@dataclass
class AIConfig:
    """Configuration for :class:`AIClient`."""

    api_key: str
    base_url: str = "https://api.openai.com/v1"
    chat_model: str = "gpt-4"
    response_model: str = "gpt-4.1-mini"
    timeout: float | None = None
    max_retries: int = 0


class AIClient:
    """Thin wrapper around the OpenAI client used throughout the codebase."""

    def __init__(self, config: AIConfig):
        self.config = config
        self._client = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=config.timeout,
            max_retries=config.max_retries,
        )

    # --- OpenAI API helpers -------------------------------------------------
    def chat_completion(
        self,
        *,
        messages: List[Dict[str, Any]],
        model: str | None = None,
        temperature: float = 0,
        **kwargs: Any,
    ):
        """Proxy to ``chat.completions.create``."""

        model = model or self.config.chat_model
        return self._client.chat.completions.create(
            model=model, messages=messages, temperature=temperature, **kwargs
        )

    def response_json(
        self,
        *,
        prompt: str,
        model: str | None = None,
        response_format: Dict[str, Any] | None = None,
        **kwargs: Any,
    ):
        """Proxy to ``responses.create`` for JSON output."""

        model = model or self.config.response_model
        return self._client.responses.create(
            model=model,
            input=prompt,
            response_format=response_format,
            **kwargs,
        )


def build_ai_client(config: AIConfig) -> AIClient:
    """Return an :class:`AIClient` instance from ``config``."""

    return AIClient(config)


_default_client: AIClient | None = None


def get_default_ai_client() -> AIClient:
    """Lazily build an :class:`AIClient` using environment configuration."""

    global _default_client
    if _default_client is None:
        from config import get_ai_config
        _default_client = build_ai_client(get_ai_config())
    return _default_client
