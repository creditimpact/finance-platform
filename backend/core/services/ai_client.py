from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping

from openai import OpenAI


logger = logging.getLogger(__name__)


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
        api_key = (config.api_key or "").strip()
        if not api_key:
            logger.error(
                "AI_CLIENT_CREDENTIAL_ERROR model=%s base_url=%s detail=missing_api_key",
                config.chat_model,
                (config.base_url or "https://api.openai.com/v1"),
            )
            raise RuntimeError(
                "AI client requires an OpenAI API key. Set OPENAI_API_KEY before starting."
            )

        base_url = (config.base_url or "").strip() or "https://api.openai.com/v1"

        logger.info(
            "AI_CLIENT_READY model=%s response_model=%s base_url=%s key_present=yes",
            config.chat_model,
            config.response_model,
            base_url,
        )

        config.api_key = api_key
        config.base_url = base_url

        self.config = config
        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url,
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
        top_p: float = 1,
        **kwargs: Any,
    ):
        """Proxy to ``chat.completions.create``."""

        model = model or self.config.chat_model

        extra_headers = kwargs.pop("extra_headers", None)
        if extra_headers:
            sanitized: Dict[str, str] = {}
            for k, v in extra_headers.items():
                cleaned = v.encode("ascii", "ignore").decode()
                if cleaned:
                    sanitized[k] = cleaned
            if sanitized:
                kwargs["extra_headers"] = sanitized

        # Enforce deterministic parameters regardless of caller provided values.
        if "frequency_penalty" in kwargs:
            kwargs.pop("frequency_penalty")
        if "presence_penalty" in kwargs:
            kwargs.pop("presence_penalty")
        if "top_p" in kwargs:
            kwargs.pop("top_p")

        frequency_penalty = 0
        presence_penalty = 0
        top_p_value = top_p

        using_tools = bool(kwargs.get("tools"))
        if not using_tools:
            kwargs.setdefault("response_format", {"type": "json_object"})

        resp = self._client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            top_p=top_p_value,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            **kwargs,
        )

        usage = getattr(resp, "usage", None)
        prompt_tokens: int | None = None
        response_tokens: int | None = None
        total_tokens: int | None = None
        if usage is not None:
            if isinstance(usage, Mapping):
                prompt_tokens = usage.get("prompt_tokens") or usage.get("input_tokens")
                response_tokens = usage.get("completion_tokens") or usage.get("output_tokens")
                total_tokens = usage.get("total_tokens")
            else:
                prompt_tokens = getattr(usage, "prompt_tokens", None) or getattr(
                    usage, "input_tokens", None
                )
                response_tokens = getattr(usage, "completion_tokens", None) or getattr(
                    usage, "output_tokens", None
                )
                total_tokens = getattr(usage, "total_tokens", None)

        choice = resp.choices[0]
        message = choice.message

        def _load_json(payload: str | None, *, context: str) -> Dict[str, Any] | None:
            if payload is None:
                return None
            try:
                parsed_payload = json.loads(payload)
                if not isinstance(parsed_payload, dict):
                    raise ValueError("Expected JSON object")
                return parsed_payload
            except Exception:  # pragma: no cover - defensive
                from backend.util.json_tools import try_fix_to_json

                fixed = try_fix_to_json(payload)
                if fixed is None:
                    logger.exception(
                        "AI_CLIENT_JSON_PARSE_FAILED context=%s", context
                    )
                    raise

                fixed_len = len(json.dumps(fixed, ensure_ascii=False))
                logger.warning(
                    "NOTE_STYLE_JSON_FIXED context=%s raw_len=%d fixed_len=%d",
                    context,
                    len(payload),
                    fixed_len,
                )
                return fixed

        content_text = getattr(message, "content", None)
        content_json = _load_json(content_text, context="content")

        tool_json = None
        tool_calls = getattr(message, "tool_calls", None)
        if content_text is None and tool_calls:
            first_call = tool_calls[0]
            arguments = getattr(getattr(first_call, "function", object()), "arguments", None)
            tool_json = _load_json(arguments, context="tool")
        else:
            arguments = None

        mode = "content" if content_text is not None else "tool"

        normalized_response = {
            "mode": mode,
            "content_json": content_json,
            "tool_json": tool_json,
            "json": content_json if content_json is not None else tool_json,
            "raw": resp,
            "openai": resp,
            "raw_content": content_text,
            "raw_tool_arguments": arguments,
        }

        if any(value is not None for value in (prompt_tokens, response_tokens, total_tokens)):
            logger.info(
                "AI_CLIENT_CHAT_USAGE model=%s prompt_tokens=%s response_tokens=%s total_tokens=%s",
                model,
                prompt_tokens if prompt_tokens is not None else "?",
                response_tokens if response_tokens is not None else "?",
                total_tokens if total_tokens is not None else "?",
            )

        return normalized_response

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
        logger.warning("Using stub AI client due to configuration error: %s", exc)
        return _StubAIClient()
