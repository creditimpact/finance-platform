from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import httpx

import backend.config as config
from backend.core.io.tags import upsert_tag
from backend.core.utils.atomic_io import atomic_write_json

from . import config as merge_config
from .ai_pack import DEFAULT_MAX_LINES


logger = logging.getLogger(__name__)


HIGHLIGHT_KEYS: tuple[str, ...] = (
    "total",
    "triggers",
    "parts",
    "matched_fields",
    "conflicts",
    "acctnum_level",
)

SYSTEM_MESSAGE = (
    "You are an expert credit tradeline merge adjudicator. Review the provided "
    "highlights and short context snippets to decide if the two accounts refer to "
    "the same underlying obligation. Treat the token '--' as missing or "
    "unknown data. Strong triggers represent decisive evidence and take "
    "priority over mid triggers; mid triggers offer supporting evidence but cannot "
    "override conflicts backed by strong signals. Creditor names may appear with "
    "aliases, abbreviations, or formatting differences—treat reasonable variants "
    "as referring to the same source when supported by other evidence. Respond "
    "ONLY with strict JSON following this schema: "
    '{"decision":"merge"|"no_merge","confidence":0..1,"reasons":[...]}. '
    "Do not add commentary or extra keys."
)


def _coerce_positive_int(value: Any, default: int) -> int:
    try:
        number = int(str(value))
    except Exception:
        return default
    return number if number > 0 else default


def _coerce_float(value: Any, default: float) -> float:
    try:
        return float(str(value))
    except Exception:
        return default


def _limit_context(lines: list[Any], limit: int) -> list[str]:
    if not lines:
        return []
    coerced = [str(item) if item is not None else "" for item in lines]
    if limit <= 0:
        return coerced
    return coerced[:limit]


def _extract_highlights(source: dict[str, Any] | None) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    if not isinstance(source, dict):
        return payload
    for key in HIGHLIGHT_KEYS:
        if key in source:
            payload[key] = source[key]
    return payload


def _build_user_message(pack: dict, max_lines: int) -> str:
    pair = pack.get("pair") or {}
    ids = pack.get("ids") or {}
    highlights = _extract_highlights(pack.get("highlights"))

    context = pack.get("context") or {}
    context_a = _limit_context(list(context.get("a") or []), max_lines)
    context_b = _limit_context(list(context.get("b") or []), max_lines)

    summary = {
        "sid": pack.get("sid", ""),
        "pair": {"a": pair.get("a"), "b": pair.get("b")},
        "account_numbers": {
            "a": ids.get("account_number_a", "--"),
            "b": ids.get("account_number_b", "--"),
        },
        "highlights": highlights,
        "context": {
            "a": context_a,
            "b": context_b,
        },
    }

    return json.dumps(summary, ensure_ascii=False, sort_keys=True)


def build_prompt_from_pack(pack: dict) -> dict[str, str]:
    limits = pack.get("limits") or {}
    default_limit = merge_config.get_ai_pack_max_lines_per_side()
    if default_limit <= 0:
        default_limit = DEFAULT_MAX_LINES
    pack_limit = _coerce_positive_int(limits.get("max_lines_per_side"), default_limit)
    max_lines = min(default_limit, pack_limit)

    user_message = _build_user_message(pack, max_lines)

    return {"system": SYSTEM_MESSAGE, "user": user_message}


def _strip_code_fences(content: str) -> str:
    text = content.strip()
    if not text.startswith("```"):
        return text

    lines = text.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _normalize_reasons(value: Any) -> list[str]:
    if not value:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if item is not None]
    return [str(value)]


def _sanitize_ai_decision(resp: Mapping[str, Any] | None, *, allow_disabled: bool) -> dict[str, Any]:
    if not isinstance(resp, Mapping):
        raise ValueError("AI decision payload must be a mapping")

    decision = str(resp.get("decision", "")).strip()
    if decision == "ai_disabled":
        if not allow_disabled:
            raise ValueError("ai_disabled decision not permitted in this context")
        return {"decision": "ai_disabled", "confidence": 0.0, "reasons": []}

    if decision not in {"merge", "no_merge"}:
        raise ValueError(f"Unsupported AI decision: {decision!r}")

    confidence_raw = resp.get("confidence", 0.0)
    confidence = _coerce_float(confidence_raw, 0.0)
    if confidence < 0.0 or confidence > 1.0:
        raise ValueError(f"Confidence must be between 0 and 1: {confidence_raw!r}")

    reasons = _normalize_reasons(resp.get("reasons"))

    return {
        "decision": decision,
        "confidence": float(confidence),
        "reasons": reasons,
    }


def _parse_ai_response(content: str) -> dict[str, Any]:
    trimmed = _strip_code_fences(content)
    data = json.loads(trimmed)
    if not isinstance(data, Mapping):
        raise ValueError("AI response JSON must be an object")
    return dict(data)


def _estimate_token_count(messages: Sequence[Mapping[str, Any]] | None) -> int:
    if not messages:
        return 0

    total_chars = 0
    for message in messages:
        if not isinstance(message, Mapping):
            continue
        content = message.get("content")
        if isinstance(content, str):
            total_chars += len(content)

    if total_chars <= 0:
        return 0

    return max(1, (total_chars + 3) // 4)


def _build_request_payload(pack: dict) -> tuple[str, dict[str, Any], dict[str, str], dict[str, Any]]:
    prompt = build_prompt_from_pack(pack)
    messages = [
        {"role": "system", "content": prompt["system"]},
        {"role": "user", "content": prompt["user"]},
    ]

    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required when AI adjudication is enabled")

    model = merge_config.get_ai_model()
    temperature = _coerce_float(
        os.getenv("AI_TEMPERATURE_DEFAULT"), getattr(config, "AI_TEMPERATURE_DEFAULT", 0.0)
    )
    max_tokens = _coerce_positive_int(
        os.getenv("AI_MAX_TOKENS"), getattr(config, "AI_MAX_TOKENS", 600)
    )

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    metadata = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    return f"{base_url}/chat/completions", payload, headers, metadata


def adjudicate_pair(pack: dict) -> dict[str, Any]:
    pair = pack.get("pair") or {}
    sid = str(pack.get("sid") or "")
    a_idx = pair.get("a")
    b_idx = pair.get("b")

    context = pack.get("context") or {}
    context_sizes = {
        "a": len(context.get("a") or []),
        "b": len(context.get("b") or []),
    }

    if not getattr(config, "ENABLE_AI_ADJUDICATOR", False):
        log_payload = {
            "sid": sid,
            "pair": {"a": a_idx, "b": b_idx},
            "reason": "disabled",
        }
        logger.info("AI_ADJUDICATOR_SKIPPED %s", json.dumps(log_payload, sort_keys=True))
        return {"decision": "ai_disabled", "confidence": 0.0, "reasons": []}

    url, payload, headers, metadata = _build_request_payload(pack)
    prompt_tokens_est = _estimate_token_count(payload.get("messages"))
    request_log = {
        "sid": sid,
        "pair": {"a": a_idx, "b": b_idx},
        "context_sizes": context_sizes,
        "model": metadata.get("model"),
        "temperature": metadata.get("temperature"),
        "max_tokens": metadata.get("max_tokens"),
        "prompt_tokens_est": prompt_tokens_est,
    }
    logger.info("AI_ADJUDICATOR_REQUEST %s", json.dumps(request_log, sort_keys=True))

    timeout_s = float(merge_config.get_ai_request_timeout())

    started = time.perf_counter()
    try:
        response = httpx.post(url, json=payload, headers=headers, timeout=timeout_s)
        response.raise_for_status()
        data = response.json()
        choices = data.get("choices") or []
        if not choices:
            raise ValueError("OpenAI response missing choices")
        message = choices[0].get("message") or {}
        content = message.get("content")
        if not isinstance(content, str):
            raise ValueError("OpenAI response missing textual content")

        parsed = _parse_ai_response(content)
        sanitized = _sanitize_ai_decision(parsed, allow_disabled=False)

        duration_ms = (time.perf_counter() - started) * 1000
        response_log = {
            "sid": sid,
            "pair": {"a": a_idx, "b": b_idx},
            "decision": sanitized["decision"],
            "confidence": sanitized["confidence"],
            "reasons_count": len(sanitized.get("reasons", [])),
            "latency_ms": round(duration_ms, 3),
        }
        logger.info("AI_ADJUDICATOR_RESPONSE %s", json.dumps(response_log, sort_keys=True))
        return sanitized
    except Exception as exc:
        duration_ms = (time.perf_counter() - started) * 1000
        error_log = {
            "sid": sid,
            "pair": {"a": a_idx, "b": b_idx},
            "error": exc.__class__.__name__,
            "latency_ms": round(duration_ms, 3),
        }
        logger.error("AI_ADJUDICATOR_ERROR %s", json.dumps(error_log, sort_keys=True))
        raise


def persist_ai_decision(
    sid: str,
    runs_root: str | os.PathLike[str],
    a_idx: int,
    b_idx: int,
    resp: Mapping[str, Any],
) -> None:
    sanitized = _sanitize_ai_decision(resp, allow_disabled=True)

    try:
        account_a = int(a_idx)
        account_b = int(b_idx)
    except Exception as exc:  # pragma: no cover - defensive
        raise ValueError("Account indices must be integers") from exc

    sid_str = str(sid)
    base = Path(runs_root) / sid_str / "cases" / "accounts"

    artifact_a = {
        "sid": sid_str,
        "pair": {"a": account_a, "b": account_b},
        "decision": sanitized["decision"],
        "confidence": sanitized["confidence"],
        "reasons": sanitized["reasons"],
    }
    artifact_b = {
        "sid": sid_str,
        "pair": {"a": account_b, "b": account_a},
        "decision": sanitized["decision"],
        "confidence": sanitized["confidence"],
        "reasons": sanitized["reasons"],
    }

    path_a = base / str(account_a) / "ai" / f"decision_pair_{account_a}_{account_b}.json"
    path_b = base / str(account_b) / "ai" / f"decision_pair_{account_b}_{account_a}.json"

    atomic_write_json(path_a.as_posix(), artifact_a, ensure_ascii=False)
    atomic_write_json(path_b.as_posix(), artifact_b, ensure_ascii=False)

    tag_a = {
        "kind": "merge_result",
        "with": account_b,
        "decision": artifact_a["decision"],
        "confidence": artifact_a["confidence"],
        "reasons": list(artifact_a["reasons"]),
        "source": "ai_adjudicator",
    }
    tag_b = {
        "kind": "merge_result",
        "with": account_a,
        "decision": artifact_b["decision"],
        "confidence": artifact_b["confidence"],
        "reasons": list(artifact_b["reasons"]),
        "source": "ai_adjudicator",
    }

    tag_path_a = base / str(account_a) / "tags.json"
    tag_path_b = base / str(account_b) / "tags.json"

    upsert_tag(tag_path_a, tag_a, ("kind", "with", "source"))
    upsert_tag(tag_path_b, tag_b, ("kind", "with", "source"))

    tag_log = {
        "sid": sid_str,
        "pair": {"a": account_a, "b": account_b},
        "decision": sanitized["decision"],
        "confidence": sanitized["confidence"],
        "reasons": list(sanitized["reasons"]),
    }
    logger.info("MERGE_V2_TAG_UPDATE %s", json.dumps(tag_log, sort_keys=True))

