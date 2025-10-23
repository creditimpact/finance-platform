"""Helpers for persisting note_style model outputs."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
import re
from typing import Any, Mapping, MutableMapping, Sequence

from backend.ai.note_style_results import store_note_style_result

from backend.core.ai.paths import NoteStyleAccountPaths


log = logging.getLogger(__name__)

_ALLOWED_TONES = {
    "neutral",
    "calm",
    "confident",
    "assertive",
    "empathetic",
    "formal",
    "conversational",
    "factual",
}

_ALLOWED_TOPICS = {
    "payment_dispute",
    "not_mine",
    "billing_error",
    "identity_theft",
    "late_fee",
    "other",
}

_ALLOWED_EMPHASIS = {
    "paid_already",
    "inaccurate_reporting",
    "identity_concerns",
    "support_request",
    "fee_waiver",
    "ownership_dispute",
    "update_requested",
    "evidence_provided",
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_existing_payload(path: Path) -> Mapping[str, Any] | None:
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError:
        log.warning("STYLE_INGEST_EXISTING_READ_FAILED path=%s", path, exc_info=True)
        return None

    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    if not lines:
        return None

    try:
        payload = json.loads(lines[-1])
    except json.JSONDecodeError:
        log.warning("STYLE_INGEST_EXISTING_PARSE_FAILED path=%s", path, exc_info=True)
        return None

    return payload if isinstance(payload, Mapping) else None


def _coerce_str(value: Any, *, preserve_case: bool = False) -> str:
    if value is None:
        return ""
    if isinstance(value, (bytes, bytearray)):
        try:
            text = value.decode("utf-8", errors="ignore")
        except Exception:
            text = ""
    else:
        text = str(value)
    text = text.strip()
    return text if preserve_case else text.lower()


def _sanitize_tone(value: Any) -> str:
    tone = _coerce_str(value)
    if tone in _ALLOWED_TONES:
        return tone
    return "neutral"


def _sanitize_topic(value: Any) -> str:
    topic = _coerce_str(value)
    if topic in _ALLOWED_TOPICS:
        return topic
    return "other"


def _sanitize_emphasis(values: Any) -> list[str]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)):
        return []
    seen: set[str] = set()
    sanitized: list[str] = []
    for entry in values:
        text = _coerce_str(entry)
        if not text or text in seen:
            continue
        if text in _ALLOWED_EMPHASIS:
            seen.add(text)
            sanitized.append(text)
    return sanitized


_RISK_FLAG_SANITIZE_PATTERN = re.compile(r"[^a-z0-9]+")


def _normalize_risk_flag(value: Any) -> str:
    text = _coerce_str(value)
    if not text:
        return ""
    sanitized = _RISK_FLAG_SANITIZE_PATTERN.sub("_", text)
    sanitized = sanitized.strip("_")
    sanitized = re.sub(r"_+", "_", sanitized)
    return sanitized


def _sanitize_risk_flags(values: Any) -> list[str]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)):
        return []
    seen: set[str] = set()
    flags: list[str] = []
    for entry in values:
        normalized = _normalize_risk_flag(entry)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        flags.append(normalized)
    return flags


def _sanitize_confidence(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    if numeric < 0:
        numeric = 0.0
    if numeric > 1:
        numeric = 1.0
    return round(numeric, 2)


def _sanitize_amount(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = _coerce_str(value, preserve_case=True)
    if not text:
        return None
    lowered = text.lower().replace("usd", "")
    cleaned = lowered.replace("$", "").replace(",", "").strip()
    try:
        return float(cleaned)
    except (TypeError, ValueError):
        return None


def _sanitize_context(context: Any) -> tuple[dict[str, Any], str, dict[str, Any]]:
    timeframe: dict[str, Any] = {"month": None, "relative": None}
    topic = "other"
    entities: dict[str, Any] = {"creditor": None, "amount": None}

    if not isinstance(context, Mapping):
        return timeframe, topic, entities

    timeframe_payload = context.get("timeframe")
    if isinstance(timeframe_payload, Mapping):
        month = _coerce_str(timeframe_payload.get("month"), preserve_case=True)
        relative = _coerce_str(
            timeframe_payload.get("relative"), preserve_case=True
        )
        timeframe["month"] = month or None
        timeframe["relative"] = relative or None

    topic = _sanitize_topic(context.get("topic"))

    entities_payload = context.get("entities")
    if isinstance(entities_payload, Mapping):
        creditor = _coerce_str(
            entities_payload.get("creditor"), preserve_case=True
        )
        entities["creditor"] = creditor or None
        entities["amount"] = _sanitize_amount(entities_payload.get("amount"))

    return timeframe, topic, entities


def _normalize_analysis(payload: Mapping[str, Any]) -> dict[str, Any]:
    tone = _sanitize_tone(payload.get("tone"))
    timeframe, topic, entities = _sanitize_context(payload.get("context_hints"))
    emphasis = _sanitize_emphasis(payload.get("emphasis"))
    confidence = _sanitize_confidence(payload.get("confidence"))
    risk_flags = _sanitize_risk_flags(payload.get("risk_flags"))

    return {
        "tone": tone,
        "context_hints": {
            "timeframe": timeframe,
            "topic": topic,
            "entities": entities,
        },
        "emphasis": emphasis,
        "confidence": confidence,
        "risk_flags": risk_flags,
    }


def _strip_code_fence(text: str) -> str:
    lines = text.strip().splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _extract_response_content(response_payload: Any) -> str:
    choices: Sequence[Any] | None = None
    if hasattr(response_payload, "choices"):
        choices = getattr(response_payload, "choices")
    elif isinstance(response_payload, Mapping):
        choices = response_payload.get("choices")  # type: ignore[assignment]

    if not isinstance(choices, Sequence) or not choices:
        raise ValueError("Model response missing choices")

    first = choices[0]
    message: Any
    if hasattr(first, "message"):
        message = getattr(first, "message")
    elif isinstance(first, Mapping):
        message = first.get("message")
    else:
        raise ValueError("Model response missing message")

    content: Any
    if hasattr(message, "content"):
        content = getattr(message, "content")
    elif isinstance(message, Mapping):
        content = message.get("content")
    else:
        raise ValueError("Model response missing content")

    if isinstance(content, str):
        text = content
    elif isinstance(content, Sequence):
        pieces: list[str] = []
        for chunk in content:
            if isinstance(chunk, str):
                pieces.append(chunk)
            elif isinstance(chunk, Mapping):
                text_piece = chunk.get("text")
                if isinstance(text_piece, str):
                    pieces.append(text_piece)
        text = "".join(pieces)
    else:
        text = str(content)

    if not text or not text.strip():
        raise ValueError("Model response missing content")

    return text.strip()


def _parse_response_payload(response_payload: Any) -> Mapping[str, Any]:
    text = _extract_response_content(response_payload)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        sanitized = _strip_code_fence(text)
        start = sanitized.find("{")
        end = sanitized.rfind("}")
        if start >= 0 and end > start:
            snippet = sanitized[start : end + 1]
            try:
                parsed = json.loads(snippet)
            except json.JSONDecodeError as exc:  # pragma: no cover - defensive
                raise ValueError("Model response is not valid JSON") from exc
        else:
            raise ValueError("Model response is not valid JSON")

    if isinstance(parsed, Mapping):
        return parsed

    raise ValueError("Model response payload must be an object")


def ingest_note_style_result(
    *,
    sid: str,
    account_id: str,
    runs_root: Path,
    account_paths: NoteStyleAccountPaths,
    pack_payload: Mapping[str, Any],
    response_payload: Any,
) -> Path:
    """Persist the normalized ``response_payload`` for ``account_id``."""

    existing_payload = _load_existing_payload(account_paths.result_file)

    prompt_salt = _coerce_str(pack_payload.get("prompt_salt"), preserve_case=True)
    if not prompt_salt and isinstance(existing_payload, Mapping):
        prompt_salt = _coerce_str(
            existing_payload.get("prompt_salt"), preserve_case=True
        )

    note_hash = _coerce_str(pack_payload.get("note_hash"), preserve_case=True)
    if not note_hash and isinstance(existing_payload, Mapping):
        note_hash = _coerce_str(
            existing_payload.get("note_hash"), preserve_case=True
        )

    parsed = _parse_response_payload(response_payload)
    analysis_payload: Mapping[str, Any]
    if isinstance(parsed.get("analysis"), Mapping):
        analysis_payload = parsed["analysis"]  # type: ignore[assignment]
    else:
        analysis_payload = parsed

    normalized_analysis = _normalize_analysis(analysis_payload)
    evaluated_at = _now_iso()

    result_payload: MutableMapping[str, Any] = {
        "sid": sid,
        "account_id": str(account_id),
        "prompt_salt": prompt_salt,
        "note_hash": note_hash,
        "analysis": normalized_analysis,
        "evaluated_at": evaluated_at,
    }

    if isinstance(existing_payload, Mapping):
        source_hash = existing_payload.get("source_hash")
        if isinstance(source_hash, str) and source_hash.strip():
            result_payload["source_hash"] = source_hash.strip()
        note_metrics = existing_payload.get("note_metrics")
        if isinstance(note_metrics, Mapping):
            result_payload["note_metrics"] = dict(note_metrics)

    log.info(
        "STYLE_INGEST_RESULT sid=%s account_id=%s prompt_salt=%s note_hash=%s",
        sid,
        account_id,
        prompt_salt,
        note_hash,
    )

    return store_note_style_result(
        sid,
        account_id,
        result_payload,
        runs_root=runs_root,
        completed_at=evaluated_at,
    )


__all__ = ["ingest_note_style_result"]
