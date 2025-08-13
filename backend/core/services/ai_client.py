from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List

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


def get_default_ai_client() -> AIClient:  # pragma: no cover - backwards compat
    """Deprecated helper that previously returned a global client.

    The application now requires explicit :class:`AIClient` injection at all
    call sites. This stub remains temporarily to surface a clearer error if
    legacy code attempts to use the old global accessor.
    """

    raise RuntimeError(
        "get_default_ai_client() has been removed. Build an AIClient via"
        " services.ai_client.build_ai_client and pass it explicitly."
    )


class _StubAIClient:
    """Fallback AI client used when configuration is missing."""

    def chat_completion(self, *a: Any, **kw: Any):  # pragma: no cover - minimal stub
        raise RuntimeError("AI client is not configured")

    def response_json(self, *a: Any, **kw: Any):  # pragma: no cover - minimal stub
        raise RuntimeError("AI client is not configured")


def get_ai_client() -> AIClient | _StubAIClient:
    """Return a configured :class:`AIClient` or a safe stub.

    If environment configuration is missing or invalid, a no-op client is
    returned which raises ``RuntimeError`` on use. This keeps the application
    dependency-injection friendly while providing a clear failure mode during
    development.
    """

    from backend.api.config import get_ai_config

    try:
        cfg = get_ai_config()
        return build_ai_client(cfg)
    except Exception as exc:  # pragma: no cover - best effort fallback
        logging.getLogger(__name__).warning(
            "Using stub AI client due to configuration error: %s", exc
        )
        return _StubAIClient()
