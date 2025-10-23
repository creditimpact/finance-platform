"""Build note_style AI stage artifacts from frontend review responses."""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import math
import os
import re
import threading
import time
import uuid
import unicodedata
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

try:  # pragma: no cover - platform dependent
    import fcntl  # type: ignore[import]
except ImportError:  # pragma: no cover - platform dependent
    fcntl = None  # type: ignore[assignment]

from backend.core.ai.paths import (
    NoteStyleAccountPaths,
    NoteStylePaths,
    ensure_note_style_account_paths,
    ensure_note_style_paths,
)
from backend.runflow.decider import record_stage


log = logging.getLogger(__name__)

_INDEX_SCHEMA_VERSION = 1
_PROMPT_PEPPER_ENV = "NOTE_STYLE_PROMPT_PEPPER"
_DEBOUNCE_MS_ENV = "NOTE_STYLE_DEBOUNCE_MS"
_DEFAULT_DEBOUNCE_MS = 750

_DEFAULT_PEPPER = "finance-note-style"
_NOTE_STYLE_MODEL = "gpt-4o-mini"
_NOTE_STYLE_SYSTEM_PROMPT = (
    "You are reviewing structured metadata that was derived from a customer's note. "
    "Respond with JSON only using the schema: "
    '{{"tone": <string>, "context_hints": {{"timeframe": {{"month": <string|null>, '
    '"relative": <string|null>}}, "topic": <string>, "entities": {{"creditor": <string|null>, '
    '"amount": <number|null>}}}}, "emphasis": [<string>...], "confidence": <float>, '
    '"risk_flags": [<string>...]}}. Confidence must be between 0 and 1. Risk flags must be '
    "lowercase snake_case strings. Do not add commentary. Prompt salt: {prompt_salt}."
)

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

_RELATIVE_TIMEFRAME_PATTERNS: dict[str, tuple[str, ...]] = {
    "last_two_months": (
        r"last\s+two\s+months",
        r"past\s+two\s+months",
        r"last\s+couple\s+of\s+months",
        r"last\s+couple\s+months",
    ),
    "last_month": (r"last\s+month", r"previous\s+month", r"past\s+month"),
    "current_month": (r"this\s+month", r"current\s+month"),
    "next_month": (r"next\s+month",),
    "last_year": (
        r"last\s+year",
        r"previous\s+year",
        r"past\s+year",
        r"past\s+twelve\s+months",
    ),
}

_MONTH_NAME_MAP = {
    "jan": "Jan",
    "january": "Jan",
    "feb": "Feb",
    "february": "Feb",
    "mar": "Mar",
    "march": "Mar",
    "apr": "Apr",
    "april": "Apr",
    "may": "May",
    "jun": "Jun",
    "june": "Jun",
    "jul": "Jul",
    "july": "Jul",
    "aug": "Aug",
    "august": "Aug",
    "sep": "Sep",
    "sept": "Sep",
    "september": "Sep",
    "oct": "Oct",
    "october": "Oct",
    "nov": "Nov",
    "november": "Nov",
    "dec": "Dec",
    "december": "Dec",
}

_KNOWN_CREDITORS = {
    "capital one": "Capital One",
    "bank of america": "Bank of America",
    "wells fargo": "Wells Fargo",
    "chase": "Chase",
    "discover": "Discover",
    "synchrony": "Synchrony",
    "citibank": "Citibank",
    "navy federal": "Navy Federal",
}

_AMOUNT_PATTERN = re.compile(r"\$?\b\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?\b")
_CREDITOR_PATTERN = re.compile(
    r"\b(?:with|from|at|to|by|for)\s+([A-Z][\w&]*(?:\s+[A-Z][\w&]*){0,3})"
)
_MONTH_PATTERN = re.compile(
    r"\b(" + "|".join(sorted(_MONTH_NAME_MAP.keys(), key=len, reverse=True)) + r")\b(?:[-/,\s]*(\d{2,4}))?",
    re.IGNORECASE,
)
_PERSONAL_DATA_PATTERNS = (
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    re.compile(r"\b\d{3}[-\.\s]\d{3}[-\.\s]\d{4}\b"),
    re.compile(r"\b\d{9}\b"),
    re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.IGNORECASE),
)

_ZERO_WIDTH_WHITESPACE = {
    ord("\u200b"): " ",  # zero width space
    ord("\u200c"): " ",  # zero width non-joiner
    ord("\u200d"): " ",  # zero width joiner
    ord("\ufeff"): " ",  # byte order mark / zero width no-break space
    ord("\u2060"): " ",  # word joiner
}


_INDEX_LOCK_POLL_INTERVAL = 0.05
_INDEX_LOCK_STALE_TIMEOUT = 30.0


@dataclass(frozen=True)
class _LoadedResponseNote:
    account_id: str
    note_sanitized: str
    source_path: Path
    source_hash: str


class NoteStyleSkip(Exception):
    """Raised when the note_style stage should soft-skip processing."""

    def __init__(self, reason: str, *, detail: str | None = None) -> None:
        super().__init__(detail or reason)
        self.reason = reason
        self.detail = detail


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _resolve_runs_root(runs_root: Path | str | None) -> Path:
    if runs_root is None:
        env_value = os.getenv("RUNS_ROOT")
        return Path(env_value) if env_value else Path("runs")
    return Path(runs_root)


def _load_json_mapping(path: Path) -> Mapping[str, Any] | None:
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError:
        log.warning("NOTE_STYLE_INDEX_READ_FAILED path=%s", path, exc_info=True)
        return None

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        log.warning("NOTE_STYLE_INDEX_PARSE_FAILED path=%s", path, exc_info=True)
        return None

    if isinstance(payload, Mapping):
        return payload

    return None


def _load_result_payload(path: Path) -> Mapping[str, Any] | None:
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError:
        log.warning("NOTE_STYLE_RESULT_READ_FAILED path=%s", path, exc_info=True)
        return None

    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    if not lines:
        return None

    try:
        payload = json.loads(lines[0])
    except json.JSONDecodeError:
        log.warning("NOTE_STYLE_RESULT_INVALID_JSON path=%s", path, exc_info=True)
        return None

    if isinstance(payload, Mapping):
        return payload

    return None


def _write_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    serialized = json.dumps(row, ensure_ascii=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + f".tmp.{uuid.uuid4().hex}")
    try:
        with tmp_path.open("w", encoding="utf-8", newline="") as handle:
            handle.write(serialized + "\n")
            handle.flush()
            try:
                os.fsync(handle.fileno())
            except OSError:
                pass
        os.replace(tmp_path, path)
    finally:
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass
    _fsync_directory(path.parent)


def _relativize(path: Path, base: Path) -> str:
    resolved_path = path.resolve()
    resolved_base = base.resolve()
    try:
        relative = resolved_path.relative_to(resolved_base)
    except ValueError:
        relative = Path(os.path.relpath(resolved_path, resolved_base))
    return str(PurePosixPath(relative))


def _unique(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        key = value.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        ordered.append(key)
    return ordered


def _pepper_bytes() -> bytes:
    value = os.getenv(_PROMPT_PEPPER_ENV)
    text = value if value else _DEFAULT_PEPPER
    return text.encode("utf-8")


def _source_hash(text: str) -> str:
    normalized = " ".join(text.split()).strip().lower()
    digest = hashlib.sha256()
    digest.update(normalized.encode("utf-8"))
    return digest.hexdigest()


def _timeframe_bucket(timeframe: Mapping[str, Any] | None) -> str:
    if not isinstance(timeframe, Mapping):
        return "none"

    relative = _normalize_text(timeframe.get("relative")).lower()
    if relative:
        return f"relative:{relative}"

    month = _normalize_text(timeframe.get("month"))
    if month:
        return f"month:{month}"

    return "none"


def _amount_band(entities: Mapping[str, Any] | None) -> str:
    amount: Any | None = None
    if isinstance(entities, Mapping):
        amount = entities.get("amount")

    try:
        value = float(amount)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return "none"

    if not math.isfinite(value):
        return "none"

    absolute = abs(value)
    if absolute == 0:
        return "zero"
    if absolute < 100:
        return "lt_100"
    if absolute < 500:
        return "100_499"
    if absolute < 1000:
        return "500_999"
    if absolute < 5000:
        return "1000_4999"
    if absolute < 10000:
        return "5000_9999"
    return "gte_10000"


def _sorted_emphasis(values: Any) -> list[str]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)):
        return []

    normalized: set[str] = set()
    for entry in values:
        text = _normalize_text(entry)
        if text:
            normalized.add(text)

    return sorted(normalized)


def _analysis_summary(
    payload: Mapping[str, Any] | None,
) -> tuple[str, str, list[str], float | None, list[str]]:
    tone = ""
    topic = ""
    emphasis: list[str] = []
    confidence_value: float | None = None
    risk_flags: list[str] = []

    if isinstance(payload, Mapping):
        tone = _normalize_text(payload.get("tone"))

        context = payload.get("context_hints")
        if isinstance(context, Mapping):
            topic = _normalize_text(context.get("topic"))

        emphasis_values = payload.get("emphasis")
        if isinstance(emphasis_values, Sequence) and not isinstance(
            emphasis_values, (str, bytes, bytearray)
        ):
            emphasis = _unique(_normalize_text(value) for value in emphasis_values)

        confidence_raw = payload.get("confidence")
        try:
            confidence_candidate = float(confidence_raw)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            confidence_candidate = None
        else:
            if math.isfinite(confidence_candidate):
                confidence_value = confidence_candidate

        risk_values = payload.get("risk_flags")
        if isinstance(risk_values, Sequence) and not isinstance(
            risk_values, (str, bytes, bytearray)
        ):
            risk_flags = _unique(_normalize_text(value) for value in risk_values)

    return tone, topic, emphasis, confidence_value, risk_flags


def _log_style_discovery(
    *,
    sid: str,
    account_id: str,
    response: PurePosixPath,
    status: str,
    note_hash: str | None = None,
    source_hash: str | None = None,
    char_len: int | None = None,
    word_len: int | None = None,
    analysis: Mapping[str, Any] | None = None,
    prompt_salt: str | None = None,
    reason: str | None = None,
) -> None:
    tone, topic, emphasis_values, confidence_value, risk_flags = _analysis_summary(analysis)
    confidence_text = ""
    if confidence_value is not None:
        confidence_text = f"{confidence_value:.2f}"

    log.info(
        "STYLE_DISCOVERY sid=%s account_id=%s response=%s status=%s note_hash=%s source_hash=%s chars=%s words=%s tone=%s topic=%s emphasis=%s confidence=%s risk_flags=%s prompt_salt=%s reason=%s",
        sid,
        account_id,
        str(response),
        status,
        note_hash or "",
        source_hash or "",
        "" if char_len is None else char_len,
        "" if word_len is None else word_len,
        tone,
        topic,
        "|".join(emphasis_values),
        confidence_text,
        "|".join(risk_flags),
        prompt_salt or "",
        reason or "",
    )


def _prompt_salt_payload(
    sid: str, account_id: str, extractor: Mapping[str, Any] | None
) -> Mapping[str, Any]:
    tone = _normalize_text((extractor or {}).get("tone") if isinstance(extractor, Mapping) else None).lower()
    if not tone:
        tone = "neutral"

    context = extractor.get("context_hints") if isinstance(extractor, Mapping) else None
    if not isinstance(context, Mapping):
        context = {}

    topic = _normalize_text(context.get("topic")).lower()
    if not topic:
        topic = "other"

    timeframe = context.get("timeframe") if isinstance(context, Mapping) else None
    if not isinstance(timeframe, Mapping):
        timeframe = {}

    entities = context.get("entities") if isinstance(context, Mapping) else None
    if not isinstance(entities, Mapping):
        entities = {}

    emphasis_sorted = _sorted_emphasis(extractor.get("emphasis") if isinstance(extractor, Mapping) else [])

    buckets = {
        "timeframe_bucket": _timeframe_bucket(timeframe),
        "amount_band": _amount_band(entities),
        "emphasis_sorted": emphasis_sorted,
    }

    return {
        "sid": str(sid),
        "account_id": str(account_id),
        "tone": tone,
        "topic": topic,
        "buckets": buckets,
    }


def _prompt_salt(sid: str, account_id: str, extractor: Mapping[str, Any] | None) -> str:
    payload = _prompt_salt_payload(sid, account_id, extractor)
    message = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hmac.new(_pepper_bytes(), message, hashlib.sha256).hexdigest()[:12]


def _note_hash(source_hash: str, length: int = 12) -> str:
    if length <= 0:
        return ""
    return source_hash[:length]


def _pack_messages(extractor: Mapping[str, Any], prompt_salt: str) -> list[dict[str, str]]:
    system_message = _NOTE_STYLE_SYSTEM_PROMPT.format(prompt_salt=prompt_salt)
    payload: dict[str, Any] = {"prompt_salt": prompt_salt}

    if isinstance(extractor, Mapping):
        payload["extractor"] = json.loads(json.dumps(extractor))

    user_content = json.dumps(payload, ensure_ascii=False, sort_keys=True)

    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_content},
    ]


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _collapse_whitespace(value: str) -> str:
    translated = value.translate(_ZERO_WIDTH_WHITESPACE)
    return " ".join(translated.split()).strip()


def _sanitize_note_value(value: Any) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        value = str(value)
    normalized = unicodedata.normalize("NFKC", value)
    return _collapse_whitespace(normalized)


def _collect_note_segments(payload: Mapping[str, Any]) -> list[str]:
    segments: list[str] = []
    seen: set[str] = set()

    def _add(value: Any) -> None:
        if isinstance(value, str):
            sanitized = _sanitize_note_value(value)
            if not sanitized or sanitized in seen:
                return
            seen.add(sanitized)
            segments.append(sanitized)
            return
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            for entry in value:
                _add(entry)

    for key in ("explanation", "note", "notes", "customer_note"):
        _add(payload.get(key))

    answers = payload.get("answers")
    if isinstance(answers, Mapping):
        for key in ("explanation", "note", "notes", "customer_note"):
            _add(answers.get(key))

    items = payload.get("items")
    if isinstance(items, Sequence):
        for entry in items:
            if not isinstance(entry, Mapping):
                _add(entry)
                continue
            for key in ("explanation", "note", "notes", "customer_note"):
                _add(entry.get(key))
            entry_answers = entry.get("answers")
            if isinstance(entry_answers, Mapping):
                for key in ("explanation", "note", "notes", "customer_note"):
                    _add(entry_answers.get(key))

    return segments


def _extract_note_text(payload: Mapping[str, Any]) -> str:
    segments = _collect_note_segments(payload)
    if not segments:
        return ""
    if len(segments) == 1:
        return segments[0]
    return _collapse_whitespace(" ".join(segments))


def _tokens(note: str) -> set[str]:
    cleaned = note.replace("/", " ").replace("-", " ")
    return {token for token in cleaned.lower().split() if token}


def _contains_phrase(text: str, *phrases: str) -> bool:
    lowered = text.lower()
    return any(phrase in lowered for phrase in phrases)


def _normalize_month_value(name: str | None, year_text: str | None) -> str | None:
    if not name:
        return None
    key = name.lower().strip(".")
    month = _MONTH_NAME_MAP.get(key)
    if not month:
        return None
    if not year_text:
        return month
    try:
        year = int(year_text)
        if year < 100:
            year += 2000 if year < 50 else 1900
        return f"{month}-{year}"
    except ValueError:
        return month


def _extract_timeframe(note: str) -> dict[str, str | None]:
    month: str | None = None
    relative: str | None = None

    match = _MONTH_PATTERN.search(note)
    if match:
        month = _normalize_month_value(match.group(1), match.group(2))

    lowered = note.lower()
    for key, patterns in _RELATIVE_TIMEFRAME_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, lowered):
                relative = key
                break
        if relative:
            break

    return {"month": month, "relative": relative}


def _extract_entities(note: str) -> tuple[dict[str, Any], list[str]]:
    entities: dict[str, Any] = {"creditor": None, "amount": None}
    lowered = note.lower()
    risk_flags: list[str] = []

    for key, display in _KNOWN_CREDITORS.items():
        if key in lowered:
            entities["creditor"] = display
            break

    if entities["creditor"] is None:
        match = _CREDITOR_PATTERN.search(note)
        if match:
            candidate = match.group(1).strip()
            words = [word for word in candidate.split() if len(word) > 1]
            if words:
                entities["creditor"] = " ".join(word.strip().title() for word in words)

    amount_match = _AMOUNT_PATTERN.search(note.replace(",", ""))
    if amount_match:
        amount_text = amount_match.group(0).replace("$", "")
        try:
            amount_value = float(amount_text)
        except ValueError:
            amount_value = None
        if amount_value is not None:
            entities["amount"] = round(amount_value, 2)
            if amount_value >= 10000:
                risk_flags.append("large_amount")

    return entities, risk_flags


def _detect_personal_data(note: str) -> bool:
    lowered = note.lower()
    if "social security" in lowered or "ssn" in lowered:
        return True
    if "date of birth" in lowered or "dob" in lowered:
        return True
    for pattern in _PERSONAL_DATA_PATTERNS:
        if pattern.search(note):
            return True
    return False


def _tone_from_note(note: str, tokens: set[str]) -> tuple[str, float, list[str]]:
    text = note.lower()
    exclamations = note.count("!")
    risk_flags: list[str] = []

    if any(word in tokens for word in {"urgent", "immediately", "asap", "now"}) or exclamations >= 2:
        tone = "assertive"
        confidence = 0.82
    elif _contains_phrase(text, "please help", "please assist", "thank you") or {
        "please",
        "help",
    }.issubset(tokens):
        tone = "empathetic"
        confidence = 0.74
    elif _contains_phrase(text, "i dispute", "this is incorrect", "not accurate") or "dispute" in tokens:
        tone = "confident"
        confidence = 0.72
    elif _contains_phrase(text, "i am requesting", "i am writing") or "sincerely" in tokens:
        tone = "formal"
        confidence = 0.7
    elif len(re.findall(r"\d", note)) >= 6 and exclamations == 0:
        tone = "factual"
        confidence = 0.68
    elif _contains_phrase(text, "just wanted") or "hey" in tokens:
        tone = "conversational"
        confidence = 0.65
    elif any(word in tokens for word in {"calm", "appreciate", "understand"}) and exclamations == 0:
        tone = "calm"
        confidence = 0.64
    else:
        tone = "neutral"
        confidence = 0.45

    if any(word in tokens for word in {"lawsuit", "court", "legal", "attorney"}):
        risk_flags.append("legal_threat")
    if exclamations >= 3:
        risk_flags.append("escalation_risk")

    if tone not in _ALLOWED_TONES:
        tone = "neutral"

    return tone, confidence, _unique(risk_flags)


def _topic_from_note(note: str, tokens: set[str]) -> tuple[str, float]:
    text = note.lower()
    if _contains_phrase(text, "identity theft", "identity fraud"):
        return "identity_theft", 0.85
    if _contains_phrase(text, "not mine", "never opened", "unauthorized"):
        return "not_mine", 0.82
    if _contains_phrase(text, "billing error", "charged the wrong", "wrong amount") or (
        "billing" in tokens and "error" in tokens
    ):
        return "billing_error", 0.8
    if _contains_phrase(text, "late fee", "late fees"):
        return "late_fee", 0.75
    if any(
        word in tokens
        for word in {"paid", "payment", "balance", "settled", "already", "dispute", "disputed", "disputing"}
    ):
        return "payment_dispute", 0.78
    return "other", 0.45


def _emphasis_from_note(note: str, tokens: set[str]) -> tuple[list[str], float, list[str]]:
    text = note.lower()
    emphasis: list[str] = []
    risk_flags: list[str] = []

    if "paid" in tokens and ("already" in tokens or _contains_phrase(text, "already paid", "paid in full")):
        emphasis.append("paid_already")
    if any(word in tokens for word in {"incorrect", "inaccurate", "error", "wrong", "mistake"}):
        emphasis.append("inaccurate_reporting")
    if _contains_phrase(text, "identity theft", "identity fraud") or "fraud" in tokens:
        emphasis.append("identity_concerns")
        risk_flags.append("identity_theft_claim")
    if _contains_phrase(text, "please help", "need assistance", "need help"):
        emphasis.append("support_request")
    if ("late" in tokens and "fee" in tokens) or _contains_phrase(text, "late fee"):
        emphasis.append("fee_waiver")
    if _contains_phrase(text, "not mine", "never opened", "unauthorized"):
        emphasis.append("ownership_dispute")
    if any(word in tokens for word in {"update", "correct", "fix", "remove", "delete"}):
        emphasis.append("update_requested")
    if any(word in tokens for word in {"attached", "documents", "proof", "evidence"}):
        emphasis.append("evidence_provided")

    filtered = [value for value in _unique(emphasis) if value in _ALLOWED_EMPHASIS]
    confidence = 0.4 if not filtered else min(0.85, 0.55 + 0.07 * len(filtered))
    return filtered, confidence, _unique(risk_flags)


def _build_extractor(note: str) -> dict[str, Any]:
    tokens = _tokens(note)
    tone, tone_confidence, tone_risks = _tone_from_note(note, tokens)
    topic, topic_confidence = _topic_from_note(note, tokens)
    emphasis_values, emphasis_confidence, emphasis_risks = _emphasis_from_note(note, tokens)
    timeframe = _extract_timeframe(note)
    entities, entity_risks = _extract_entities(note)

    risk_flags = set(tone_risks) | set(emphasis_risks) | set(entity_risks)
    if _detect_personal_data(note):
        risk_flags.add("personal_data")

    confidence = 0.3
    if tone != "neutral":
        confidence += min(0.25, tone_confidence * 0.3)
    if topic != "other":
        confidence += min(0.2, topic_confidence * 0.2)
    if emphasis_values:
        confidence += min(0.25, emphasis_confidence * 0.2 + 0.05 * len(emphasis_values))
    if timeframe.get("month") or timeframe.get("relative"):
        confidence += 0.05
    if entities.get("amount") is not None or entities.get("creditor"):
        confidence += 0.05

    confidence = round(min(confidence, 0.95), 2)

    if confidence < 0.5:
        tone = "neutral"
        emphasis_values = []
        topic = "other"

    if tone not in _ALLOWED_TONES:
        tone = "neutral"
    if topic not in _ALLOWED_TOPICS:
        topic = "other"

    emphasis_values = [value for value in emphasis_values if value in _ALLOWED_EMPHASIS]

    if "personal_data" in risk_flags:
        entities = {"creditor": None, "amount": None}

    extractor = {
        "tone": tone,
        "context_hints": {
            "timeframe": {
                "month": timeframe.get("month"),
                "relative": timeframe.get("relative"),
            },
            "topic": topic,
            "entities": {
                "creditor": entities.get("creditor"),
                "amount": entities.get("amount"),
            },
        },
        "emphasis": emphasis_values,
        "confidence": confidence,
        "risk_flags": sorted(risk_flags),
    }

    return extractor


def _load_response_note(account_id: str, response_path: Path) -> _LoadedResponseNote:
    try:
        raw = response_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise NoteStyleSkip("missing_response") from None
    except OSError:
        log.warning("NOTE_STYLE_RESPONSE_READ_FAILED path=%s", response_path, exc_info=True)
        raise NoteStyleSkip("response_read_failed") from None

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        log.warning("NOTE_STYLE_RESPONSE_INVALID_JSON path=%s", response_path, exc_info=True)
        raise NoteStyleSkip("invalid_response") from None

    if not isinstance(payload, Mapping):
        raise NoteStyleSkip("invalid_response")

    note_text = _extract_note_text(payload)
    if not note_text:
        raise NoteStyleSkip("empty_note")

    source_hash = _source_hash(note_text)
    return _LoadedResponseNote(
        account_id=str(account_id),
        note_sanitized=note_text,
        source_path=response_path,
        source_hash=source_hash,
    )


def _index_items(payload: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(payload, Mapping):
        return []
    packs = payload.get("packs")
    if isinstance(packs, Sequence):
        return [dict(entry) for entry in packs if isinstance(entry, Mapping)]
    items = payload.get("items")
    if isinstance(items, Sequence):
        return [dict(entry) for entry in items if isinstance(entry, Mapping)]
    return []


def _serialize_entry(
    *,
    sid: str,
    account_id: str,
    paths: NoteStylePaths,
    account_paths: NoteStyleAccountPaths,
    note_hash: str,
    status: str,
    timestamp: str,
) -> dict[str, Any]:
    return {
        "account_id": account_id,
        "pack": _relativize(account_paths.pack_file, paths.base),
        "result": _relativize(account_paths.result_file, paths.base),
        "lines": 1,
        "built_at": timestamp,
        "status": status,
        "source_hash": note_hash,
    }


def _compute_totals(items: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    total = 0
    completed = 0
    failed = 0
    for entry in items:
        status = _normalize_text(entry.get("status")).lower()
        if status in {"", "skipped"}:
            continue
        total += 1
        if status in {"completed", "success", "built"}:
            completed += 1
        elif status in {"failed", "error"}:
            failed += 1
    if total == 0 and completed == 0 and failed == 0:
        return {"total": 0, "completed": 0, "failed": 0}
    return {"total": total, "completed": completed, "failed": failed}


def _coerce_int(value: Any) -> int | None:
    try:
        if value is None or isinstance(value, bool):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _fsync_directory(directory: Path) -> None:
    try:
        fd = os.open(str(directory), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        try:
            os.close(fd)
        except OSError:
            pass


@contextmanager
def _index_lock(index_path: Path) -> Iterator[None]:
    """Serialize index writers to avoid concurrent clobbering."""

    index_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = index_path.with_suffix(index_path.suffix + ".lock")

    if fcntl is not None:  # pragma: no branch - preferred path on POSIX
        with lock_path.open("a+") as handle:
            while True:
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                    break
                except InterruptedError:
                    continue
            try:
                yield
            finally:
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                finally:
                    try:
                        lock_path.unlink()
                    except FileNotFoundError:
                        pass
        return

    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError:
            try:
                stats = lock_path.stat()
            except FileNotFoundError:
                continue
            if (time.time() - stats.st_mtime) > _INDEX_LOCK_STALE_TIMEOUT:
                try:
                    os.unlink(lock_path)
                except FileNotFoundError:
                    continue
                continue
            time.sleep(_INDEX_LOCK_POLL_INTERVAL)

    try:
        os.close(fd)
    except OSError:
        pass

    try:
        yield
    finally:
        try:
            os.unlink(lock_path)
        except FileNotFoundError:
            pass


def _atomic_write_index(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + f".tmp.{uuid.uuid4().hex}")
    try:
        with tmp_path.open("w", encoding="utf-8", newline="") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, indent=2))
            handle.flush()
            try:
                os.fsync(handle.fileno())
            except OSError:
                pass
        os.replace(tmp_path, path)
    finally:
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass
    _fsync_directory(path.parent)


def _write_index(
    *,
    sid: str,
    paths: NoteStylePaths,
    items: Sequence[Mapping[str, Any]],
) -> None:
    document = {
        "schema_version": _INDEX_SCHEMA_VERSION,
        "sid": sid,
        "root": ".",
        "packs_dir": _relativize(paths.packs_dir, paths.base),
        "results_dir": _relativize(paths.results_dir, paths.base),
        "packs": list(items),
    }
    _atomic_write_index(paths.index_file, document)


def _remove_account_artifacts(account_paths: NoteStyleAccountPaths) -> None:
    for path in (account_paths.pack_file, account_paths.result_file):
        try:
            path.unlink()
        except FileNotFoundError:
            continue
        except OSError:
            log.warning("NOTE_STYLE_ARTIFACT_REMOVE_FAILED path=%s", path, exc_info=True)


def _update_index_for_account(
    *,
    sid: str,
    paths: NoteStylePaths,
    account_id: str,
    entry: Mapping[str, Any] | None,
) -> dict[str, int]:
    index_path = paths.index_file
    replaced_flag = False
    created_flag = False
    removed_flag = False
    removed_entry: Mapping[str, Any] | None = None
    with _index_lock(index_path):
        existing = _load_json_mapping(index_path)
        items = _index_items(existing)

        normalized_account = str(account_id)
        rewritten: list[dict[str, Any]] = []
        for item in items:
            if str(item.get("account_id")) == normalized_account:
                if entry is not None:
                    rewritten.append(dict(entry))
                    replaced_flag = True
                else:
                    removed_flag = True
                    removed_entry = dict(item)
                continue
            rewritten.append(item)

        if entry is not None and not replaced_flag:
            rewritten.append(dict(entry))
            created_flag = True

        rewritten.sort(key=lambda item: str(item.get("account_id", "")))
        _write_index(sid=sid, paths=paths, items=rewritten)
        totals = _compute_totals(rewritten)

    if entry is None:
        if removed_flag:
            action = "removed"
        else:
            action = "noop"
        status_text = "removed" if removed_flag else ""
        pack_value = str((removed_entry or {}).get("pack") or "")
        result_value = str((removed_entry or {}).get("result") or "")
    else:
        action = "updated" if replaced_flag else "created" if created_flag else "noop"
        status_text = _normalize_text(entry.get("status")) if isinstance(entry, Mapping) else ""
        pack_value = str(entry.get("pack") or "")
        result_value = str(entry.get("result") or "")

    index_relative = _relativize(paths.index_file, paths.base)
    source_hash_value = ""
    if entry is not None and isinstance(entry, Mapping):
        source_hash_value = str(entry.get("source_hash") or "")
    elif removed_entry is not None:
        source_hash_value = str(removed_entry.get("source_hash") or "")
    log.info(
        "STYLE_INDEX_UPDATED sid=%s account_id=%s action=%s status=%s packs_total=%s packs_completed=%s packs_failed=%s index=%s pack=%s result=%s source_hash=%s",
        sid,
        account_id,
        action,
        status_text,
        totals.get("total", 0),
        totals.get("completed", 0),
        totals.get("failed", 0),
        index_relative,
        pack_value,
        result_value,
        source_hash_value,
    )

    return totals


def _note_style_index_progress(index_path: Path) -> tuple[int, int, int]:
    document = _load_json_mapping(index_path)
    if not isinstance(document, Mapping):
        return (0, 0, 0)

    entries: Sequence[Mapping[str, Any]] = ()
    packs_payload = document.get("packs")
    if isinstance(packs_payload, Sequence):
        entries = [entry for entry in packs_payload if isinstance(entry, Mapping)]
    else:
        items_payload = document.get("items")
        if isinstance(items_payload, Sequence):
            entries = [entry for entry in items_payload if isinstance(entry, Mapping)]

    total = 0
    completed = 0
    failed = 0

    for entry in entries:
        status_text = _normalize_text(entry.get("status")).lower()
        if status_text in {"", "skipped"}:
            continue
        total += 1
        if status_text == "completed":
            completed += 1
        elif status_text in {"failed", "error"}:
            failed += 1

    if not entries:
        totals_payload = document.get("totals")
        if isinstance(totals_payload, Mapping):
            total = _coerce_int(totals_payload.get("total")) or 0
            completed = _coerce_int(totals_payload.get("completed")) or 0
            failed = _coerce_int(totals_payload.get("failed")) or 0

    return (max(total, 0), max(completed, 0), max(failed, 0))


def _record_stage_progress(
    *, sid: str, runs_root: Path, totals: Mapping[str, int], index_path: Path
) -> None:
    packs_total, packs_completed, packs_failed = _note_style_index_progress(index_path)

    if packs_failed > 0 and packs_total > 0:
        status: str = "error"
    elif packs_total == 0 or packs_completed == packs_total:
        status = "success"
    else:
        status = "built"

    empty_ok = packs_total == 0

    counts = {"packs_total": packs_total}
    metrics = {"packs_total": packs_total}
    results = {
        "results_total": packs_total,
        "completed": packs_completed,
        "failed": packs_failed,
    }

    record_stage(
        sid,
        "note_style",
        status=status,
        counts=counts,
        empty_ok=empty_ok,
        metrics=metrics,
        results=results,
        runs_root=runs_root,
    )


def build_note_style_pack_for_account(
    sid: str, account_id: str, *, runs_root: Path | str | None = None
) -> Mapping[str, Any]:
    runs_root_path = _resolve_runs_root(runs_root)
    run_dir = runs_root_path / sid
    account_id_str = str(account_id)
    response_rel = PurePosixPath(
        f"runs/{sid}/frontend/review/responses/{account_id}.result.json"
    )
    response_path = (
        run_dir
        / "frontend"
        / "review"
        / "responses"
        / f"{account_id}.result.json"
    )

    paths = ensure_note_style_paths(runs_root_path, sid, create=True)
    account_paths = ensure_note_style_account_paths(paths, account_id, create=True)

    try:
        loaded_note = _load_response_note(account_id, response_path)
    except NoteStyleSkip as exc:
        reason = exc.reason or "empty_note"
        totals = _update_index_for_account(sid=sid, paths=paths, account_id=account_id, entry=None)
        _remove_account_artifacts(account_paths)
        _record_stage_progress(
            sid=sid, runs_root=runs_root_path, totals=totals, index_path=paths.index_file
        )
        _log_style_discovery(
            sid=sid,
            account_id=account_id_str,
            response=response_rel,
            status="skipped",
            reason=reason,
        )
        return {
            "status": "skipped",
            "reason": reason,
            "packs_total": totals.get("total", 0),
            "packs_completed": totals.get("completed", 0),
        }

    note_text = loaded_note.note_sanitized
    source_hash = loaded_note.source_hash
    note_hash = _note_hash(source_hash)
    char_len = len(note_text)
    word_len = len(note_text.split())
    existing_index = _load_json_mapping(paths.index_file)
    items = _index_items(existing_index)
    existing_entry: Mapping[str, Any] | None = None
    for item in items:
        if str(item.get("account_id")) == str(account_id):
            existing_entry = item
            break

    existing_result = _load_result_payload(account_paths.result_file)
    if (
        existing_entry is not None
        and str(existing_entry.get("source_hash")) == note_hash
        and isinstance(existing_result, Mapping)
        and str(existing_result.get("source_hash")) == source_hash
    ):
        totals = _compute_totals(items)
        _record_stage_progress(
            sid=sid,
            runs_root=runs_root_path,
            totals=totals,
            index_path=paths.index_file,
        )
        existing_analysis: Mapping[str, Any] | None = None
        if isinstance(existing_result, Mapping):
            analysis_payload = existing_result.get("analysis")
            extractor_payload = existing_result.get("extractor")
            if isinstance(analysis_payload, Mapping):
                existing_analysis = analysis_payload
            elif isinstance(extractor_payload, Mapping):
                existing_analysis = extractor_payload
        prompt_salt_existing = (
            str(existing_result.get("prompt_salt")) if isinstance(existing_result, Mapping) else ""
        )
        _log_style_discovery(
            sid=sid,
            account_id=account_id_str,
            response=response_rel,
            status="unchanged",
            note_hash=note_hash,
            source_hash=source_hash,
            char_len=char_len,
            word_len=word_len,
            analysis=existing_analysis,
            prompt_salt=prompt_salt_existing,
        )
        return {
            "status": "unchanged",
            "packs_total": totals.get("total", 0),
            "packs_completed": totals.get("completed", 0),
        }

    extractor = _build_extractor(note_text)
    prompt_salt = _prompt_salt(sid, str(account_id), extractor)
    _log_style_discovery(
        sid=sid,
        account_id=account_id_str,
        response=response_rel,
        status="ready",
        note_hash=note_hash,
        source_hash=source_hash,
        char_len=char_len,
        word_len=word_len,
        analysis=extractor,
        prompt_salt=prompt_salt,
    )
    timestamp = _now_iso()

    pack_payload = {
        "sid": sid,
        "account_id": str(account_id),
        "source_response_path": str(response_rel),
        "note_hash": note_hash,
        "model": _NOTE_STYLE_MODEL,
        "extractor": extractor,
        "messages": _pack_messages(extractor, prompt_salt),
        "built_at": timestamp,
    }
    result_payload = {
        "sid": sid,
        "account_id": str(account_id),
        "analysis": extractor,
        "extractor": extractor,
        "prompt_salt": prompt_salt,
        "source_hash": source_hash,
        "note_hash": note_hash,
        "evaluated_at": timestamp,
    }

    _write_jsonl(account_paths.pack_file, pack_payload)
    _write_jsonl(account_paths.result_file, result_payload)

    tone_value, topic_value, emphasis_values, confidence_value, risk_values = _analysis_summary(extractor)
    confidence_text = ""
    if confidence_value is not None:
        confidence_text = f"{confidence_value:.2f}"
    pack_relative = _relativize(account_paths.pack_file, paths.base)
    result_relative = _relativize(account_paths.result_file, paths.base)
    log.info(
        "STYLE_PACK_BUILT sid=%s account_id=%s pack=%s result=%s tone=%s topic=%s emphasis=%s confidence=%s risk_flags=%s prompt_salt=%s note_hash=%s source_hash=%s model=%s",
        sid,
        account_id_str,
        pack_relative,
        result_relative,
        tone_value,
        topic_value,
        "|".join(emphasis_values),
        confidence_text,
        "|".join(risk_values),
        prompt_salt,
        note_hash,
        source_hash,
        _NOTE_STYLE_MODEL,
    )

    entry = _serialize_entry(
        sid=sid,
        account_id=str(account_id),
        paths=paths,
        account_paths=account_paths,
        note_hash=note_hash,
        status="built",
        timestamp=timestamp,
    )

    totals = _update_index_for_account(
        sid=sid, paths=paths, account_id=str(account_id), entry=entry
    )
    _record_stage_progress(
        sid=sid, runs_root=runs_root_path, totals=totals, index_path=paths.index_file
    )

    return {
        "status": "completed",
        "packs_total": totals.get("total", 0),
        "packs_completed": totals.get("completed", 0),
        "prompt_salt": prompt_salt,
        "note_hash": note_hash,
    }


@dataclass
class _DebounceEntry:
    timer: threading.Timer


_DEBOUNCE_LOCK = threading.Lock()
_PENDING: dict[tuple[str, str], _DebounceEntry] = {}


def _schedule_timer(delay: float, fn: Callable[[], None]) -> threading.Timer:
    timer = threading.Timer(delay, fn)
    timer.daemon = True
    timer.start()
    return timer


def _debounce_delay_seconds() -> float:
    raw = os.getenv(_DEBOUNCE_MS_ENV)
    if raw is None:
        return _DEFAULT_DEBOUNCE_MS / 1000.0
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return _DEFAULT_DEBOUNCE_MS / 1000.0
    if value <= 0:
        return 0.0
    return value / 1000.0


def schedule_note_style_refresh(
    sid: str, account_id: str, *, runs_root: Path | str | None = None
) -> None:
    delay = _debounce_delay_seconds()

    def _run() -> None:
        try:
            build_note_style_pack_for_account(sid, account_id, runs_root=runs_root)
        except Exception:  # pragma: no cover - defensive
            log.exception(
                "NOTE_STYLE_BUILD_FAILED sid=%s account_id=%s", sid, account_id
            )
        finally:
            with _DEBOUNCE_LOCK:
                _PENDING.pop((sid, account_id), None)

    if delay <= 0:
        _run()
        return

    with _DEBOUNCE_LOCK:
        existing = _PENDING.pop((sid, account_id), None)
        if existing is not None:
            try:
                existing.timer.cancel()
            except Exception:
                pass
        timer = _schedule_timer(delay, _run)
        _PENDING[(sid, account_id)] = _DebounceEntry(timer=timer)


__all__ = [
    "build_note_style_pack_for_account",
    "schedule_note_style_refresh",
]

