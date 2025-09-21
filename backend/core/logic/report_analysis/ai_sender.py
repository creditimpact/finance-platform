"""Send merge adjudication packs to the AI adjudicator service."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Iterable, Mapping, MutableMapping, Sequence

import httpx

from backend.core.io.tags import upsert_tag


DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_TIMEOUT = 30.0

# Retry configuration – one initial attempt plus three retries using this schedule.
RETRY_BACKOFF_SECONDS: Sequence[float] = (1.0, 3.0, 7.0)
MAX_RETRIES = len(RETRY_BACKOFF_SECONDS)

ALLOWED_DECISIONS = {"merge", "same_debt", "different"}


@dataclass(frozen=True)
class AISenderConfig:
    """Configuration required to contact the chat completion API."""

    base_url: str
    api_key: str
    model: str
    timeout: float


@dataclass
class SendOutcome:
    """Result of attempting to adjudicate a single pack."""

    success: bool
    attempts: int
    decision: str | None = None
    reason: str | None = None
    error_kind: str | None = None
    error_message: str | None = None


def _bool_from_env(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    return default


def is_enabled() -> bool:
    """Return whether AI adjudication is enabled via configuration."""

    env_value = os.getenv("ENABLE_AI_ADJUDICATOR")
    enabled = _bool_from_env(env_value, default=False)
    if env_value is not None:
        return enabled

    try:  # pragma: no cover - defensive fallback when module is absent
        import backend.config as backend_config  # type: ignore

        return bool(getattr(backend_config, "ENABLE_AI_ADJUDICATOR", False))
    except Exception:  # pragma: no cover - optional dependency
        return False


def _coerce_positive_float(raw: str | None, default: float) -> float:
    if raw is None:
        return default
    try:
        value = float(str(raw).strip())
    except Exception:
        return default
    if value <= 0:
        return default
    return value


def load_config_from_env() -> AISenderConfig:
    """Build :class:`AISenderConfig` from environment variables."""

    base_url = os.getenv("OPENAI_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required when sending AI merge packs")

    model = os.getenv("AI_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL
    timeout = _coerce_positive_float(os.getenv("AI_REQUEST_TIMEOUT"), DEFAULT_TIMEOUT)

    return AISenderConfig(base_url=base_url, api_key=api_key, model=model, timeout=timeout)


def _format_url(base_url: str) -> str:
    return f"{base_url.rstrip('/')}/chat/completions"


def _default_http_request(
    url: str,
    payload: Mapping[str, Any],
    headers: Mapping[str, str],
    timeout: float,
) -> httpx.Response:
    return httpx.post(url, json=dict(payload), headers=dict(headers), timeout=timeout)


def _strip_code_fences(text: str) -> str:
    trimmed = text.strip()
    if not trimmed.startswith("```"):
        return trimmed

    lines = [line for line in trimmed.splitlines()]
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _parse_model_payload(content: str) -> MutableMapping[str, Any]:
    try:
        data = json.loads(_strip_code_fences(content))
    except json.JSONDecodeError as exc:
        raise ValueError("Model response must be valid JSON") from exc
    if not isinstance(data, MutableMapping):
        raise ValueError("Model response JSON must be an object")
    return data


def _sanitize_decision(payload: Mapping[str, Any]) -> tuple[str, str]:
    raw_decision = payload.get("decision")
    decision = str(raw_decision).strip().lower()
    if decision not in ALLOWED_DECISIONS:
        raise ValueError(f"Unsupported decision: {raw_decision!r}")

    reason_raw = payload.get("reason")
    if reason_raw is None:
        raise ValueError("Model response missing reason")
    reason = str(reason_raw).strip()
    if not reason:
        raise ValueError("Model response reason must be non-empty")

    return decision, reason


def send_single_attempt(
    pack: Mapping[str, Any],
    config: AISenderConfig,
    *,
    request: Callable[[str, Mapping[str, Any], Mapping[str, str], float], httpx.Response] | None = None,
) -> tuple[str, str]:
    """Send ``pack`` once and return the decision and reason."""

    messages = pack.get("messages")
    if not isinstance(messages, Sequence):
        raise ValueError("Pack is missing messages payload")

    url = _format_url(config.base_url)
    payload = {
        "model": config.model,
        "messages": list(messages),
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
    }
    headers = {
        "Authorization": f"Bearer {config.api_key}",
        "Content-Type": "application/json",
    }

    sender = request or _default_http_request
    response = sender(url, payload, headers, config.timeout)
    response.raise_for_status()
    data = response.json()
    choices = data.get("choices") if isinstance(data, Mapping) else None
    if not choices:
        raise ValueError("Model response missing choices")
    message = choices[0].get("message") if isinstance(choices[0], Mapping) else None
    if not isinstance(message, Mapping):
        raise ValueError("Model response missing message")
    content = message.get("content")
    if not isinstance(content, str):
        raise ValueError("Model response missing textual content")

    parsed = _parse_model_payload(content)
    return _sanitize_decision(parsed)


LogCallback = Callable[[str, Mapping[str, Any]], None]


def process_pack(
    pack: Mapping[str, Any],
    config: AISenderConfig,
    *,
    request: Callable[[str, Mapping[str, Any], Mapping[str, str], float], httpx.Response] | None = None,
    sleep: Callable[[float], None] = time.sleep,
    log: LogCallback | None = None,
) -> SendOutcome:
    """Attempt to adjudicate ``pack`` using retry logic."""

    attempts = 0
    last_error: Exception | None = None
    max_attempts = 1 + MAX_RETRIES

    while attempts < max_attempts:
        attempts += 1
        if log is not None:
            log(
                "REQUEST",
                {
                    "attempt": attempts,
                    "max_attempts": max_attempts,
                },
            )

        try:
            decision, reason = send_single_attempt(pack, config, request=request)
            if log is not None:
                log(
                    "RESPONSE",
                    {
                        "attempt": attempts,
                        "decision": decision,
                    },
                )
            return SendOutcome(success=True, attempts=attempts, decision=decision, reason=reason)
        except Exception as exc:  # pragma: no cover - diverse error sources
            last_error = exc
            will_retry = attempts <= MAX_RETRIES
            if log is not None and will_retry:
                payload = {
                    "attempt": attempts,
                    "error": exc.__class__.__name__,
                    "will_retry": True,
                }
                log("ERROR", payload)

            if not will_retry:
                break

            delay = RETRY_BACKOFF_SECONDS[min(attempts - 1, len(RETRY_BACKOFF_SECONDS) - 1)]
            if log is not None:
                log(
                    "RETRY",
                    {
                        "attempt": attempts,
                        "delay_seconds": delay,
                    },
                )
            sleep(delay)

    error_kind = last_error.__class__.__name__ if last_error is not None else "UnknownError"
    error_message = str(last_error) if last_error is not None else "unknown"
    if log is not None and last_error is not None:
        log(
            "ERROR",
            {
                "attempt": attempts,
                "error": error_kind,
                "will_retry": False,
                "final": True,
            },
        )
    return SendOutcome(
        success=False,
        attempts=attempts,
        decision=None,
        reason=None,
        error_kind=error_kind,
        error_message=error_message,
    )


def isoformat_timestamp(dt: datetime | None = None) -> str:
    """Return a UTC ISO-8601 timestamp without fractional seconds."""

    if dt is None:
        dt = datetime.now(timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt.isoformat(timespec="seconds").replace("+00:00", "Z")


def _account_tags_dir(runs_root: os.PathLike[str] | str, sid: str) -> str:
    base = os.fspath(runs_root)
    return os.path.join(base, sid, "cases", "accounts")


def _ensure_int(value: Any, label: str) -> int:
    try:
        return int(value)
    except Exception as exc:  # pragma: no cover - defensive
        raise ValueError(f"{label} must be an integer") from exc


def write_decision_tags(
    runs_root: os.PathLike[str] | str,
    sid: str,
    a_idx: Any,
    b_idx: Any,
    decision: str,
    reason: str,
    at: str,
) -> None:
    """Write symmetric ai_decision (and optional same_debt) tags for the pair."""

    account_a = _ensure_int(a_idx, "a_idx")
    account_b = _ensure_int(b_idx, "b_idx")

    base = _account_tags_dir(runs_root, sid)

    def _tag_payload(source: int, other: int) -> dict[str, Any]:
        return {
            "kind": "ai_decision",
            "tag": "ai_decision",
            "source": "ai_adjudicator",
            "with": other,
            "decision": decision,
            "reason": reason,
            "at": at,
        }

    for source_idx, other_idx in ((account_a, account_b), (account_b, account_a)):
        tag_path = os.path.join(base, str(source_idx), "tags.json")
        upsert_tag(tag_path, _tag_payload(source_idx, other_idx), ("kind", "with", "source"))
        if decision == "same_debt":
            same_debt_tag = {
                "kind": "same_debt_pair",
                "with": other_idx,
                "source": "ai_adjudicator",
                "at": at,
            }
            upsert_tag(tag_path, same_debt_tag, ("kind", "with", "source"))


def write_error_tags(
    runs_root: os.PathLike[str] | str,
    sid: str,
    a_idx: Any,
    b_idx: Any,
    error_kind: str,
    message: str,
    at: str,
) -> None:
    """Write symmetric ai_error tags for the pair."""

    account_a = _ensure_int(a_idx, "a_idx")
    account_b = _ensure_int(b_idx, "b_idx")

    base = _account_tags_dir(runs_root, sid)

    def _payload(other: int) -> dict[str, Any]:
        return {
            "kind": "ai_error",
            "with": other,
            "source": "ai_adjudicator",
            "error_kind": error_kind,
            "message": message,
            "at": at,
        }

    for source_idx, other_idx in ((account_a, account_b), (account_b, account_a)):
        tag_path = os.path.join(base, str(source_idx), "tags.json")
        upsert_tag(tag_path, _payload(other_idx), ("kind", "with", "source"))


__all__ = [
    "AISenderConfig",
    "SendOutcome",
    "ALLOWED_DECISIONS",
    "RETRY_BACKOFF_SECONDS",
    "MAX_RETRIES",
    "is_enabled",
    "load_config_from_env",
    "send_single_attempt",
    "process_pack",
    "isoformat_timestamp",
    "write_decision_tags",
    "write_error_tags",
]

