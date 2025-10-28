"""Helpers for persisting note_style model results."""

from __future__ import annotations

import copy
import json
import logging
import math
import os
import hashlib
import re
import time
import uuid
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence

from backend.ai.manifest import (
    ensure_note_style_section,
    update_note_style_stage_status,
)
from backend.ai.note_style.io import note_style_stage_view
from backend.ai.note_style_logging import (
    append_note_style_warning,
    log_structured_event,
)
from backend.core.ai.paths import (
    NoteStyleAccountPaths,
    NoteStylePaths,
    ensure_note_style_account_paths,
    ensure_note_style_paths,
)
from backend.core.runflow import runflow_barriers_refresh
from backend.runflow.decider import (
    record_stage,
    reconcile_umbrella_barriers,
    refresh_note_style_stage_from_results,
)

log = logging.getLogger(__name__)


_SHORT_NOTE_WORD_LIMIT = 12
_SHORT_NOTE_CHAR_LIMIT = 90
_MAX_ANALYSIS_LIST_ITEMS = 6
_UNSUPPORTED_CLAIM_KEYWORDS = (
    "legal",  # general legal language
    "lawsuit",
    "attorney",
    "court",
    "legal claim",
    "sue",
    "filing",
    "owe me",
    "owe us",
    "owed me",
    "owed us",
)
_NO_DOCUMENT_PATTERNS = (
    "no documents",
    "no document",
    "no documentation",
    "without documents",
    "without documentation",
    "no proof",
    "without proof",
    "no evidence",
    "without evidence",
    "don't have documents",
    "do not have documents",
    "don't have proof",
    "do not have proof",
)


def _resolve_runs_root(runs_root: Path | str | None) -> Path:
    if runs_root is None:
        env_value = os.getenv("RUNS_ROOT")
        return Path(env_value) if env_value else Path("runs")
    return Path(runs_root)


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _relative_to_base(path: Path, base: Path) -> str:
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _structured_repr(payload: Any) -> str:
    try:
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)
    except Exception:
        return str(payload)


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


def _atomic_write_jsonl(path: Path, payload: Mapping[str, Any]) -> int:
    serialized = json.dumps(payload, ensure_ascii=False)
    data = (serialized + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + f".tmp.{uuid.uuid4().hex}")
    try:
        with tmp_path.open("wb") as handle:
            handle.write(data)
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
    return len(data)


def _to_snake_case(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    cleaned = re.sub(r"[^0-9A-Za-z]+", " ", text)
    cleaned = cleaned.strip().lower()
    return re.sub(r"\s+", "_", cleaned)


_DATE_FULL_FORMATS: tuple[str, ...] = (
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%Y.%m.%d",
    "%m/%d/%Y",
    "%m-%d-%Y",
    "%m.%d.%Y",
    "%m/%d/%y",
    "%m-%d-%y",
    "%m.%d.%y",
    "%B %d, %Y",
    "%b %d, %Y",
    "%B %d %Y",
    "%b %d %Y",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%SZ",
)

_DATE_MONTH_FORMATS: tuple[str, ...] = (
    "%Y-%m",
    "%Y/%m",
    "%Y.%m",
    "%B %Y",
    "%b %Y",
    "%B-%Y",
    "%b-%Y",
)


def _normalize_date_field(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    text = str(value).strip()
    if not text:
        return None
    for fmt in _DATE_FULL_FORMATS:
        try:
            parsed = datetime.strptime(text, fmt)
        except ValueError:
            continue
        return parsed.date().isoformat()
    if text.endswith("Z"):
        try:
            parsed = datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ")
        except ValueError:
            parsed = None
        if parsed is not None:
            return parsed.date().isoformat()
    try:
        parsed_iso = datetime.fromisoformat(text)
    except ValueError:
        parsed_iso = None
    if parsed_iso is not None:
        return parsed_iso.date().isoformat()
    for fmt in _DATE_MONTH_FORMATS:
        try:
            parsed = datetime.strptime(text, fmt)
        except ValueError:
            continue
        return datetime(parsed.year, parsed.month, 1).date().isoformat()
    return None


def _normalize_timestamp_field(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value.astimezone(timezone.utc) if value.tzinfo else value.replace(tzinfo=timezone.utc)
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    if isinstance(value, date):
        dt = datetime(value.year, value.month, value.day, tzinfo=timezone.utc)
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    text = str(value).strip()
    if not text:
        return None
    candidates = [
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%d %H:%M:%S",
        "%Y/%m/%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
    ]
    for fmt in candidates:
        try:
            parsed = datetime.strptime(text, fmt)
        except ValueError:
            continue
        parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.strftime("%Y-%m-%dT%H:%M:%SZ")
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    else:
        parsed = parsed.astimezone(timezone.utc)
    return parsed.strftime("%Y-%m-%dT%H:%M:%SZ")


def _ensure_mutable_mapping(value: Mapping[str, Any]) -> MutableMapping[str, Any]:
    if isinstance(value, MutableMapping):
        return value
    return dict(value)


def _normalize_risk_flags_list(values: Any) -> list[str]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)):
        return []
    normalized: list[str] = []
    seen: set[str] = set()
    for value in values:
        candidate = _to_snake_case(value)
        if not candidate or candidate in seen:
            continue
        normalized.append(candidate)
        seen.add(candidate)
    return normalized


def _coerce_sequence(values: Any) -> list[Any]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)):
        return []
    return list(values)


def _normalize_emphasis_values(values: Any) -> list[str]:
    raw_items = _coerce_sequence(values)
    normalized: list[str] = []
    seen: set[str] = set()
    for item in raw_items:
        candidate = _to_snake_case(item)
        if not candidate or candidate in seen:
            continue
        normalized.append(candidate)
        seen.add(candidate)
    return normalized


def _coerce_positive_int(value: Any) -> int | None:
    try:
        numeric = int(value)
    except (TypeError, ValueError):
        try:
            numeric = int(float(value))
        except (TypeError, ValueError):
            return None
    if numeric < 0:
        return None
    return numeric


def _coerce_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(numeric) or math.isinf(numeric):
        return None
    return numeric


def _normalize_context_section(context: MutableMapping[str, Any]) -> None:
    if "risk_flags" in context:
        context["risk_flags"] = _normalize_risk_flags_list(context.get("risk_flags"))
    topic_value = context.get("topic")
    if topic_value is None:
        context["topic"] = None
    else:
        topic_normalized = _to_snake_case(topic_value)
        context["topic"] = topic_normalized or None
    timeframe_payload = context.get("timeframe")
    if isinstance(timeframe_payload, Mapping):
        timeframe_mutable = _ensure_mutable_mapping(timeframe_payload)
        month_normalized = _normalize_date_field(timeframe_mutable.get("month"))
        timeframe_mutable["month"] = month_normalized
        relative_value = timeframe_mutable.get("relative")
        if relative_value is None:
            timeframe_mutable["relative"] = None
        else:
            relative_normalized = _to_snake_case(relative_value)
            timeframe_mutable["relative"] = relative_normalized or None
        context["timeframe"] = timeframe_mutable


def _normalize_analysis_section(analysis: MutableMapping[str, Any]) -> None:
    if "risk_flags" in analysis:
        analysis["risk_flags"] = _normalize_risk_flags_list(analysis.get("risk_flags"))

    tone_payload = analysis.get("tone")
    if isinstance(tone_payload, Mapping):
        tone_mutable = _ensure_mutable_mapping(tone_payload)
        if "risk_flags" in tone_mutable:
            tone_mutable["risk_flags"] = _normalize_risk_flags_list(
                tone_mutable.get("risk_flags")
            )
        analysis["tone"] = tone_mutable

    context_payload = analysis.get("context_hints")
    if isinstance(context_payload, Mapping):
        context_mutable = _ensure_mutable_mapping(context_payload)
        _normalize_context_section(context_mutable)
        analysis["context_hints"] = context_mutable

    emphasis_payload = analysis.get("emphasis")
    if isinstance(emphasis_payload, Mapping):
        emphasis_mutable = _ensure_mutable_mapping(emphasis_payload)
        if "risk_flags" in emphasis_mutable:
            emphasis_mutable["risk_flags"] = _normalize_risk_flags_list(
                emphasis_mutable.get("risk_flags")
            )
        analysis["emphasis"] = emphasis_mutable


def _normalize_result_payload(payload: Mapping[str, Any]) -> MutableMapping[str, Any]:
    normalized: MutableMapping[str, Any] = {
        "sid": str(payload.get("sid") or "").strip(),
        "account_id": str(payload.get("account_id") or "").strip(),
    }

    evaluated_at = _normalize_timestamp_field(payload.get("evaluated_at"))
    if evaluated_at is not None:
        normalized["evaluated_at"] = evaluated_at

    analysis_payload = payload.get("analysis")
    if isinstance(analysis_payload, Mapping):
        analysis_mutable = _ensure_mutable_mapping(analysis_payload)
        _normalize_analysis_section(analysis_mutable)
        normalized["analysis"] = analysis_mutable
    elif analysis_payload is None:
        normalized["analysis"] = None

    metrics_payload = payload.get("note_metrics")
    if isinstance(metrics_payload, Mapping):
        sanitized_metrics = _sanitize_note_metrics(metrics_payload)
        if sanitized_metrics is not None:
            normalized["note_metrics"] = sanitized_metrics

    return normalized


def _relativize(path: Path, base: Path) -> str:
    resolved_path = path.resolve()
    resolved_base = base.resolve()
    try:
        relative = resolved_path.relative_to(resolved_base)
    except ValueError:
        relative = Path(os.path.relpath(resolved_path, resolved_base))
    return relative.as_posix()


def _compute_totals(entries: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    total = 0
    completed = 0
    failed = 0
    for entry in entries:
        status = str(entry.get("status") or "").strip().lower()
        if status in {"", "skipped", "skipped_low_signal"}:
            continue
        total += 1
        if status in {"completed", "success"}:
            completed += 1
        elif status in {"failed", "error"}:
            failed += 1
    return {"total": total, "completed": completed, "failed": failed}


def _note_style_log_path(paths: NoteStylePaths) -> Path:
    return getattr(paths, "log_file", paths.base / "logs.txt")


def _load_pack_payload_for_logging(pack_path: Path) -> Mapping[str, Any] | None:
    try:
        raw = pack_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError:
        log.warning("NOTE_STYLE_PACK_READ_FAILED path=%s", pack_path, exc_info=True)
        return None

    for line in raw.splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            log.warning("NOTE_STYLE_PACK_PARSE_FAILED path=%s", pack_path, exc_info=True)
            return None
        if isinstance(payload, Mapping):
            return payload
        break
    return None


def _looks_like_analysis_payload(candidate: Mapping[str, Any]) -> bool:
    if not isinstance(candidate, Mapping):
        return False

    tone_value = candidate.get("tone")
    if not isinstance(tone_value, str) or not tone_value.strip():
        return False

    context_payload = candidate.get("context_hints")
    if not isinstance(context_payload, Mapping):
        return False

    timeframe_payload = context_payload.get("timeframe")
    if not isinstance(timeframe_payload, Mapping):
        return False
    if "month" not in timeframe_payload or "relative" not in timeframe_payload:
        return False

    entities_payload = context_payload.get("entities")
    if not isinstance(entities_payload, Mapping):
        return False
    if "creditor" not in entities_payload or "amount" not in entities_payload:
        return False

    emphasis_payload = candidate.get("emphasis")
    if not (
        isinstance(emphasis_payload, Sequence)
        and not isinstance(emphasis_payload, (str, bytes, bytearray))
    ):
        return False

    confidence_value = _coerce_float(candidate.get("confidence"))
    if confidence_value is None:
        return False

    risk_flags_payload = candidate.get("risk_flags")
    if risk_flags_payload is None:
        return False
    if not (
        isinstance(risk_flags_payload, Sequence)
        and not isinstance(risk_flags_payload, (str, bytes, bytearray))
    ):
        return False

    return True


def _result_file_contains_valid_analysis(result_path: Path | None) -> bool:
    if result_path is None or not result_path.exists():
        return False

    try:
        with result_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                text = line.strip()
                if not text:
                    continue
                try:
                    payload = json.loads(text)
                except json.JSONDecodeError:
                    log.warning(
                        "NOTE_STYLE_RESULT_PARSE_FAILED path=%s",
                        result_path,
                        exc_info=True,
                    )
                    return False
                if not isinstance(payload, Mapping):
                    return False
                if _looks_like_analysis_payload(payload):
                    return True
                analysis_payload = payload.get("analysis")
                if isinstance(analysis_payload, Mapping) and _looks_like_analysis_payload(
                    analysis_payload
                ):
                    return True
                return False
    except FileNotFoundError:
        return False
    except OSError:
        log.warning(
            "NOTE_STYLE_RESULT_READ_FAILED path=%s", result_path, exc_info=True
        )
        return False

    return False


def _load_pack_note_metrics(pack_path: Path) -> Mapping[str, Any] | None:
    payload = _load_pack_payload_for_logging(pack_path)
    if isinstance(payload, Mapping):
        sanitized = _sanitize_note_metrics(payload.get("note_metrics"))
        if sanitized is not None:
            return sanitized
    return None


def _load_pack_note_text(pack_path: Path) -> str | None:
    payload = _load_pack_payload_for_logging(pack_path)
    if not isinstance(payload, Mapping):
        return None
    note_candidate = payload.get("note_text")
    if not isinstance(note_candidate, str):
        context = payload.get("context")
        if isinstance(context, Mapping):
            context_note = context.get("note_text")
            if isinstance(context_note, str):
                note_candidate = context_note
    if isinstance(note_candidate, str):
        return note_candidate
    messages = payload.get("messages")
    if not isinstance(messages, Sequence):
        return None
    for message in messages:
        if not isinstance(message, Mapping):
            continue
        role = str(message.get("role") or "").strip().lower()
        if role != "user":
            continue
        content = message.get("content")
        if isinstance(content, Mapping):
            note_text = content.get("note_text")
            if isinstance(note_text, str):
                return note_text
        elif isinstance(content, str):
            return content
    return None


def _is_short_note_metrics(note_metrics: Mapping[str, Any] | None) -> bool:
    if not isinstance(note_metrics, Mapping):
        return False
    word_len = _coerce_positive_int(note_metrics.get("word_len"))
    if word_len is not None and word_len <= _SHORT_NOTE_WORD_LIMIT:
        return True
    char_len = _coerce_positive_int(note_metrics.get("char_len"))
    if char_len is not None and char_len <= _SHORT_NOTE_CHAR_LIMIT:
        return True
    return False


def _enforce_confidence_cap(analysis: MutableMapping[str, Any], limit: float) -> None:
    confidence_value = _coerce_float(analysis.get("confidence"))
    if confidence_value is None:
        return
    capped = min(max(confidence_value, 0.0), limit)
    analysis["confidence"] = round(capped, 2)


def _append_risk_flag(analysis: MutableMapping[str, Any], flag: str) -> None:
    existing = _normalize_risk_flags_list(analysis.get("risk_flags"))
    addition = _normalize_risk_flags_list([flag])
    for entry in addition:
        if entry not in existing:
            existing.append(entry)
    if existing:
        analysis["risk_flags"] = existing


def _detect_unsupported_claim(note_text: str) -> bool:
    normalized = " ".join(note_text.lower().split())
    if not normalized:
        return False
    if not any(keyword in normalized for keyword in _UNSUPPORTED_CLAIM_KEYWORDS):
        return False
    return any(pattern in normalized for pattern in _NO_DOCUMENT_PATTERNS)


def _sanitize_note_metrics(candidate: Any) -> dict[str, int] | None:
    if not isinstance(candidate, Mapping):
        return None
    char_len = _coerce_positive_int(candidate.get("char_len"))
    word_len = _coerce_positive_int(candidate.get("word_len"))
    if char_len is None or word_len is None:
        return None
    return {"char_len": char_len, "word_len": word_len}


def _resolve_note_metrics(
    payload: MutableMapping[str, Any],
    *,
    pack_path: Path,
) -> Mapping[str, Any] | None:
    note_metrics_candidate = payload.get("note_metrics")
    if isinstance(note_metrics_candidate, Mapping):
        sanitized = _sanitize_note_metrics(note_metrics_candidate)
        if sanitized is not None:
            payload["note_metrics"] = sanitized
            return sanitized
    pack_metrics = _load_pack_note_metrics(pack_path)
    if isinstance(pack_metrics, Mapping):
        sanitized_pack = _sanitize_note_metrics(pack_metrics)
        if sanitized_pack is not None:
            payload["note_metrics"] = sanitized_pack
            return sanitized_pack
    note_text = _load_pack_note_text(pack_path)
    if isinstance(note_text, str):
        fallback_metrics = {
            "char_len": len(note_text),
            "word_len": len(note_text.split()),
        }
        sanitized_fallback = _sanitize_note_metrics(fallback_metrics)
        if sanitized_fallback is not None:
            payload["note_metrics"] = sanitized_fallback
            return sanitized_fallback
    return None


def _apply_result_enhancements(
    payload: MutableMapping[str, Any],
    *,
    note_metrics: Mapping[str, Any] | None,
    pack_path: Path,
) -> None:
    analysis_payload = payload.get("analysis")
    if not isinstance(analysis_payload, Mapping):
        return
    analysis_mutable = _ensure_mutable_mapping(analysis_payload)
    if _is_short_note_metrics(note_metrics):
        _enforce_confidence_cap(analysis_mutable, 0.5)
    note_text = _load_pack_note_text(pack_path)
    if note_text and _detect_unsupported_claim(note_text):
        _append_risk_flag(analysis_mutable, "unsupported_claim")
    payload["analysis"] = analysis_mutable


def _guard_result_analysis(
    *,
    sid: str,
    account_id: str,
    paths: NoteStylePaths,
    payload: MutableMapping[str, Any],
) -> None:
    analysis_payload = payload.get("analysis")
    if analysis_payload is None:
        return

    if not isinstance(analysis_payload, Mapping):
        detail = "analysis_not_mapping"
        log.warning(
            "NOTE_STYLE_RESULT_GUARD_FAILED sid=%s account_id=%s issues=%s",
            sid,
            account_id,
            detail,
        )
        append_note_style_warning(
            _note_style_log_path(paths),
            f"sid={sid} account_id={account_id} guard_failed issues={detail}",
        )
        raise ValueError("analysis payload must be an object")

    allowed_keys = {"tone", "context_hints", "emphasis", "confidence", "risk_flags"}
    sanitized: dict[str, Any] = {}
    adjustments: list[str] = []
    fatal: list[str] = []

    extra_keys = sorted(
        key for key in analysis_payload.keys() if key not in allowed_keys
    )
    if extra_keys:
        adjustments.append(
            "extra_keys_removed=" + ":".join(extra_keys)
        )

    tone_value = analysis_payload.get("tone")
    if isinstance(tone_value, Mapping):
        tone_text = str(tone_value.get("value") or "").strip()
    else:
        tone_text = str(tone_value or "").strip()
    if tone_text:
        sanitized["tone"] = tone_text
    else:
        fatal.append("tone")

    context_payload = analysis_payload.get("context_hints")
    if isinstance(context_payload, Mapping):
        context_sanitized: dict[str, Any] = {
            "timeframe": {"month": None, "relative": None},
            "topic": None,
            "entities": {"creditor": None, "amount": None},
        }

        timeframe_payload = context_payload.get("timeframe")
        if isinstance(timeframe_payload, Mapping):
            month_raw = timeframe_payload.get("month")
            month_value: str | None = None
            if month_raw is not None:
                month_text = str(month_raw).strip()
                normalized_month = _normalize_date_field(month_text)
                if normalized_month:
                    month_value = normalized_month
                    if normalized_month != month_text:
                        adjustments.append("context.timeframe.month_normalized")
                else:
                    month_value = month_text or None
                    if month_value != month_text:
                        adjustments.append("context.timeframe.month_cleared")
            context_sanitized["timeframe"]["month"] = month_value

            relative_raw = timeframe_payload.get("relative")
            if relative_raw is None:
                relative_value = None
            else:
                relative_value = _to_snake_case(relative_raw) or None
                original_relative = str(relative_raw).strip()
                if relative_value is None and original_relative:
                    adjustments.append("context.timeframe.relative_cleared")
                elif relative_value is not None and relative_value != original_relative:
                    adjustments.append("context.timeframe.relative_normalized")
            context_sanitized["timeframe"]["relative"] = relative_value
        elif timeframe_payload is not None:
            adjustments.append("context.timeframe_reset")

        topic_raw = context_payload.get("topic")
        topic_value = None
        if topic_raw is not None:
            topic_value = _to_snake_case(topic_raw) or None
            original_topic = str(topic_raw).strip()
            if topic_value is None and original_topic:
                adjustments.append("context.topic_cleared")
            elif topic_value is not None and topic_value != original_topic:
                adjustments.append("context.topic_normalized")
        context_sanitized["topic"] = topic_value

        entities_payload = context_payload.get("entities")
        if isinstance(entities_payload, Mapping):
            creditor_raw = entities_payload.get("creditor")
            if creditor_raw is None:
                creditor_value = None
            else:
                creditor_text = str(creditor_raw).strip()
                creditor_value = creditor_text or None
                if creditor_value is None and creditor_text:
                    adjustments.append("context.entities.creditor_cleared")
            amount_raw = entities_payload.get("amount")
            amount_value: float | None = None
            if amount_raw not in (None, ""):
                numeric = _coerce_float(amount_raw)
                if numeric is None:
                    adjustments.append("context.entities.amount_cleared")
                else:
                    amount_value = round(numeric, 2)
                    if amount_value != numeric:
                        adjustments.append("context.entities.amount_rounded")
            context_sanitized["entities"]["creditor"] = creditor_value
            context_sanitized["entities"]["amount"] = amount_value
        elif entities_payload is not None:
            adjustments.append("context.entities_reset")

        sanitized["context_hints"] = context_sanitized
    else:
        fatal.append("context_hints")

    emphasis_payload = analysis_payload.get("emphasis")
    emphasis_source: Any
    if isinstance(emphasis_payload, Mapping):
        emphasis_source = emphasis_payload.get("values")
    else:
        emphasis_source = emphasis_payload
    emphasis_original = _coerce_sequence(emphasis_source)
    emphasis_values = _normalize_emphasis_values(emphasis_source)
    if len(emphasis_values) < len(emphasis_original):
        adjustments.append("emphasis_normalized")
    if len(emphasis_values) > _MAX_ANALYSIS_LIST_ITEMS:
        adjustments.append("emphasis_truncated")
    sanitized["emphasis"] = emphasis_values[:_MAX_ANALYSIS_LIST_ITEMS]

    confidence_raw = analysis_payload.get("confidence")
    confidence_value = _coerce_float(confidence_raw)
    if confidence_value is None:
        fatal.append("confidence")
    else:
        clamped = max(0.0, min(1.0, confidence_value))
        if clamped != confidence_value:
            adjustments.append("confidence_clamped")
        sanitized["confidence"] = round(clamped, 3)

    risk_flags_raw = analysis_payload.get("risk_flags")
    risk_flags_original = _coerce_sequence(risk_flags_raw)
    risk_flags_values = _normalize_risk_flags_list(risk_flags_raw)
    if len(risk_flags_values) < len(risk_flags_original):
        adjustments.append("risk_flags_normalized")
    if len(risk_flags_values) > _MAX_ANALYSIS_LIST_ITEMS:
        adjustments.append("risk_flags_truncated")
    sanitized["risk_flags"] = risk_flags_values[:_MAX_ANALYSIS_LIST_ITEMS]

    if fatal:
        detail = ",".join(sorted(dict.fromkeys(fatal)))
        log.warning(
            "NOTE_STYLE_RESULT_GUARD_FAILED sid=%s account_id=%s issues=%s",
            sid,
            account_id,
            detail,
        )
        append_note_style_warning(
            _note_style_log_path(paths),
            f"sid={sid} account_id={account_id} guard_failed issues={detail}",
        )
        raise ValueError("analysis payload missing required fields")

    if adjustments:
        detail = ",".join(sorted(dict.fromkeys(adjustments)))
        log.warning(
            "NOTE_STYLE_RESULT_GUARD_ADJUSTED sid=%s account_id=%s adjustments=%s",
            sid,
            account_id,
            detail,
        )
        append_note_style_warning(
            _note_style_log_path(paths),
            f"sid={sid} account_id={account_id} guard_adjustments={detail}",
        )

    payload["analysis"] = sanitized


def _validate_result_payload(
    *,
    sid: str,
    account_id: str,
    paths: NoteStylePaths,
    account_paths: NoteStyleAccountPaths,
    payload: Mapping[str, Any],
) -> None:
    missing_fields: list[str] = []
    analysis_payload = payload.get("analysis")
    sid_value = str(payload.get("sid") or "").strip()
    if not sid_value:
        missing_fields.append("sid")
    account_value = str(payload.get("account_id") or "").strip()
    if not account_value:
        missing_fields.append("account_id")
    evaluated_at_value = str(payload.get("evaluated_at") or "").strip()
    if not evaluated_at_value:
        missing_fields.append("evaluated_at")

    note_metrics = payload.get("note_metrics")
    sanitized_metrics = (
        _sanitize_note_metrics(note_metrics) if isinstance(note_metrics, Mapping) else None
    )
    if sanitized_metrics is None:
        missing_fields.extend([
            "note_metrics.char_len",
            "note_metrics.word_len",
        ])

    if isinstance(analysis_payload, Mapping):
        tone = str(analysis_payload.get("tone") or "").strip()
        if not tone:
            missing_fields.append("tone")

        topic_value = ""
        context_payload = analysis_payload.get("context_hints")
        if isinstance(context_payload, Mapping):
            topic_value = str(context_payload.get("topic") or "").strip()
            timeframe = context_payload.get("timeframe")
            if isinstance(timeframe, Mapping):
                month_value = timeframe.get("month")
                if month_value is not None and not isinstance(month_value, str):
                    missing_fields.append("timeframe.month")
                relative_value = timeframe.get("relative")
                if relative_value is not None and not isinstance(relative_value, str):
                    missing_fields.append("timeframe.relative")
            else:
                missing_fields.append("timeframe")

            entities_payload = context_payload.get("entities")
            if isinstance(entities_payload, Mapping):
                creditor_value = entities_payload.get("creditor")
                if creditor_value is not None and not isinstance(creditor_value, str):
                    missing_fields.append("entities.creditor")
                amount_value = entities_payload.get("amount")
                if amount_value is not None and not isinstance(amount_value, (int, float)):
                    missing_fields.append("entities.amount")
            else:
                missing_fields.append("entities")
        else:
            missing_fields.append("context_hints")

        if not topic_value:
            missing_fields.append("topic")

        if analysis_payload.get("confidence") is None:
            missing_fields.append("confidence")

        emphasis = analysis_payload.get("emphasis")
        if emphasis is None:
            missing_fields.append("emphasis")

        risk_flags = analysis_payload.get("risk_flags")
        if risk_flags is None:
            missing_fields.append("risk_flags")
    else:
        missing_fields.append("analysis")

    missing_artifacts: list[str] = []
    if not account_paths.pack_file.exists():
        missing_artifacts.append("pack")
    if not account_paths.result_file.exists():
        missing_artifacts.append("result")

    if not missing_fields and not missing_artifacts:
        return

    field_text = ",".join(sorted(dict.fromkeys(missing_fields))) if missing_fields else ""
    artifact_text = ",".join(sorted(dict.fromkeys(missing_artifacts))) if missing_artifacts else ""

    parts: list[str] = []
    if field_text:
        parts.append(f"missing_fields={field_text}")
    if artifact_text:
        parts.append(f"missing_artifacts={artifact_text}")

    detail = " ".join(parts) if parts else "validation_warning"
    raw_hint = account_paths.result_raw_file
    if not raw_hint.exists():
        json_candidate = raw_hint.with_suffix(".json")
        if json_candidate.exists():
            raw_hint = json_candidate

    log.warning(
        "NOTE_STYLE_RESULT_INVALID sid=%s account_id=%s reason=%s raw_path=%s",
        sid,
        account_id,
        detail,
        raw_hint.resolve().as_posix(),
    )
    append_note_style_warning(
        _note_style_log_path(paths),
        f"sid={sid} account_id={account_id} {detail}".strip(),
    )


class NoteStyleIndexWriter:
    """Maintain the note_style index file for a run."""

    def __init__(self, *, sid: str, paths: NoteStylePaths) -> None:
        self.sid = str(sid)
        self._paths = paths
        self._index_path = paths.index_file
        self._index_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _load_document(self) -> MutableMapping[str, Any]:
        try:
            raw = self._index_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return {}
        except OSError:
            log.warning(
                "NOTE_STYLE_INDEX_READ_FAILED sid=%s path=%s",
                self.sid,
                self._index_path,
                exc_info=True,
            )
            return {}
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            log.warning(
                "NOTE_STYLE_INDEX_PARSE_FAILED sid=%s path=%s",
                self.sid,
                self._index_path,
                exc_info=True,
            )
            return {}
        if isinstance(payload, MutableMapping):
            return dict(payload)
        if isinstance(payload, Mapping):
            return dict(payload)
        return {}

    def _extract_entries(
        self, document: MutableMapping[str, Any]
    ) -> tuple[str, list[Mapping[str, Any]]]:
        for key in ("packs", "items"):
            container = document.get(key)
            if isinstance(container, Sequence):
                entries = [entry for entry in container if isinstance(entry, Mapping)]
                document[key] = list(entries)
                return key, list(entries)
        document["packs"] = []
        return "packs", []

    def _atomic_write_index(self, document: Mapping[str, Any]) -> None:
        tmp_path = self._index_path.with_suffix(
            self._index_path.suffix + f".tmp.{uuid.uuid4().hex}"
        )
        try:
            with tmp_path.open("w", encoding="utf-8", newline="") as handle:
                json.dump(document, handle, ensure_ascii=False, indent=2)
                handle.flush()
                try:
                    os.fsync(handle.fileno())
                except OSError:
                    pass
            os.replace(tmp_path, self._index_path)
        finally:
            try:
                tmp_path.unlink()
            except FileNotFoundError:
                pass
        _fsync_directory(self._index_path.parent)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def mark_completed(
        self,
        account_id: str,
        *,
        pack_path: Path | None,
        result_path: Path | None,
        completed_at: str | None = None,
    ) -> tuple[Mapping[str, Any] | None, dict[str, int], int, bool]:
        document = self._load_document()
        key, entries = self._extract_entries(document)

        timestamp = completed_at or _now_iso()
        normalized_account = str(account_id)
        analysis_valid = _result_file_contains_valid_analysis(result_path)

        rewritten: list[dict[str, Any]] = []
        updated_entry: dict[str, Any] | None = None
        for entry in entries:
            entry_payload = dict(entry)
            if str(entry_payload.get("account_id") or "") == normalized_account:
                if analysis_valid:
                    entry_payload["status"] = "completed"
                    entry_payload["completed_at"] = timestamp
                else:
                    status_text = str(entry_payload.get("status") or "").strip().lower()
                    if status_text in {"completed", "success"}:
                        entry_payload["status"] = "pending"
                    entry_payload.pop("completed_at", None)
                if result_path is not None:
                    entry_payload["result_path"] = _relativize(
                        result_path, self._paths.base
                    )
                else:
                    entry_payload.setdefault("result_path", "")
                if pack_path is not None:
                    entry_payload.setdefault(
                        "pack", _relativize(pack_path, self._paths.base)
                    )
                entry_payload.pop("error", None)
                updated_entry = entry_payload
            rewritten.append(entry_payload)

        if updated_entry is None:
            entry_payload = {"account_id": normalized_account}
            if analysis_valid:
                entry_payload["status"] = "completed"
                entry_payload["completed_at"] = timestamp
            else:
                entry_payload["status"] = "pending"
            if pack_path is not None:
                entry_payload["pack"] = _relativize(pack_path, self._paths.base)
            if result_path is not None:
                entry_payload["result_path"] = _relativize(
                    result_path, self._paths.base
                )
            else:
                entry_payload["result_path"] = ""
            rewritten.append(entry_payload)
            updated_entry = entry_payload

        rewritten.sort(key=lambda item: str(item.get("account_id") or ""))
        document[key] = rewritten
        document["totals"] = _compute_totals(rewritten)
        skipped_count = sum(
            1
            for entry in rewritten
            if str(entry.get("status") or "").strip().lower()
            in {"skipped", "skipped_low_signal"}
        )

        self._atomic_write_index(document)
        totals = document["totals"]
        index_relative = _relative_to_base(self._index_path, self._paths.base)
        status_text = str(updated_entry.get("status") or "") if updated_entry else ""
        pack_value = str(updated_entry.get("pack") or "") if updated_entry else ""
        result_value = (
            str(updated_entry.get("result_path") or "") if updated_entry else ""
        )
        log.info(
            "NOTE_STYLE_INDEX_UPDATED sid=%s account_id=%s action=completed status=%s packs_total=%s packs_completed=%s packs_failed=%s skipped=%s index=%s pack=%s result_path=%s",
            self.sid,
            normalized_account,
            status_text,
            totals.get("total", 0),
            totals.get("completed", 0),
            totals.get("failed", 0),
            skipped_count,
            index_relative,
            pack_value,
            result_value,
        )
        if result_path is not None and not analysis_valid:
            log.warning(
                "NOTE_STYLE_RESULT_ANALYSIS_MISSING sid=%s account_id=%s path=%s",
                self.sid,
                normalized_account,
                _relative_to_base(result_path, self._paths.base),
            )
        return updated_entry, totals, skipped_count, analysis_valid

    def mark_failed(
        self,
        account_id: str,
        *,
        pack_path: Path | None,
        error: str | None = None,
    ) -> tuple[Mapping[str, Any], dict[str, int]]:
        document = self._load_document()
        key, entries = self._extract_entries(document)

        timestamp = _now_iso()
        normalized_account = str(account_id)

        rewritten: list[dict[str, Any]] = []
        updated_entry: dict[str, Any] | None = None
        for entry in entries:
            entry_payload = dict(entry)
            if str(entry_payload.get("account_id") or "") == normalized_account:
                entry_payload["status"] = "failed"
                entry_payload["failed_at"] = timestamp
                if error:
                    entry_payload["error"] = str(error)
                else:
                    entry_payload.pop("error", None)
                if pack_path is not None:
                    entry_payload.setdefault(
                        "pack", _relativize(pack_path, self._paths.base)
                    )
                entry_payload.setdefault("result_path", "")
                updated_entry = entry_payload
            rewritten.append(entry_payload)

        if updated_entry is None:
            entry_payload = {
                "account_id": normalized_account,
                "status": "failed",
                "failed_at": timestamp,
                "result_path": "",
            }
            if pack_path is not None:
                entry_payload["pack"] = _relativize(pack_path, self._paths.base)
            if error:
                entry_payload["error"] = str(error)
            rewritten.append(entry_payload)
            updated_entry = entry_payload

        rewritten.sort(key=lambda item: str(item.get("account_id") or ""))
        document[key] = rewritten
        document["totals"] = _compute_totals(rewritten)
        skipped_count = sum(
            1
            for entry in rewritten
            if str(entry.get("status") or "").strip().lower()
            in {"skipped", "skipped_low_signal"}
        )

        self._atomic_write_index(document)
        totals = document["totals"]

        index_relative = _relative_to_base(self._index_path, self._paths.base)
        status_text = str(updated_entry.get("status") or "") if updated_entry else ""
        pack_value = str(updated_entry.get("pack") or "") if updated_entry else ""
        error_value = str(updated_entry.get("error") or "") if updated_entry else ""
        log.info(
            "NOTE_STYLE_INDEX_UPDATED sid=%s account_id=%s action=failed status=%s packs_total=%s packs_completed=%s packs_failed=%s skipped=%s index=%s pack=%s error=%s",
            self.sid,
            normalized_account,
            status_text,
            totals.get("total", 0),
            totals.get("completed", 0),
            totals.get("failed", 0),
            skipped_count,
            index_relative,
            pack_value,
            error_value,
        )
        return updated_entry, totals, skipped_count


def _refresh_after_index_update(
    *,
    sid: str,
    account_id: str,
    runs_root_path: Path,
    paths: NoteStylePaths,
    updated_entry: Mapping[str, Any] | None,
    totals: Mapping[str, Any],
    skipped_count: int,
    completed_at_override: str | None = None,
) -> None:
    results_override: tuple[int, int, int] | None = None
    try:
        results_override = refresh_note_style_stage_from_results(
            sid, runs_root=runs_root_path
        )
    except Exception:  # pragma: no cover - defensive logging
        log.warning(
            "NOTE_STYLE_STAGE_REFRESH_FAILED sid=%s", sid, exc_info=True
        )
    else:
        status_text = (
            str(updated_entry.get("status") or "")
            if isinstance(updated_entry, Mapping)
            else ""
        )
        log.info(
            "NOTE_STYLE_STAGE_REFRESH_DETAIL sid=%s account_id=%s stage_status=%s packs_total=%s results_completed=%s results_failed=%s",
            sid,
            account_id,
            status_text,
            totals.get("total", 0),
            totals.get("completed", 0),
            totals.get("failed", 0),
        )
        try:
            barrier_state = reconcile_umbrella_barriers(
                sid, runs_root=runs_root_path
            )
        except Exception:  # pragma: no cover - defensive logging
            log.warning(
                "NOTE_STYLE_BARRIERS_RECONCILE_FAILED sid=%s", sid, exc_info=True
            )
        else:
            log.info(
                "[Runflow] Umbrella barriers: sid=%s stage=note_style state=%s",
                sid,
                _structured_repr(barrier_state),
            )

    view = note_style_stage_view(sid, runs_root=runs_root_path)

    packs_total = view.total_expected
    packs_completed = view.completed_total
    packs_failed = view.failed_total
    stage_status = view.state
    ready = stage_status == "success"
    empty_ok = packs_total == 0

    counts = {"packs_total": packs_total}
    metrics = {"packs_total": packs_total}
    results_counts = {
        "results_total": packs_total,
        "completed": packs_completed,
        "failed": packs_failed,
    }

    log.info(
        "NOTE_STYLE_REFRESH sid=%s ready=%s total=%s completed=%s failed=%s skipped=%s",
        sid,
        ready,
        packs_total,
        packs_completed,
        packs_failed,
        skipped_count,
    )
    log_structured_event(
        "NOTE_STYLE_REFRESH",
        logger=log,
        sid=sid,
        ready=ready,
        status=stage_status,
        packs_total=packs_total,
        packs_completed=packs_completed,
        packs_failed=packs_failed,
        packs_skipped=skipped_count,
    )

    completed_timestamp: str | None
    if completed_at_override is not None:
        completed_timestamp = completed_at_override
    elif view.is_terminal:
        completed_timestamp = _now_iso()
    else:
        completed_timestamp = None

    sent_flag = view.is_terminal or completed_at_override is not None

    try:
        update_note_style_stage_status(
            sid,
            runs_root=runs_root_path,
            built=view.built_complete,
            sent=sent_flag,
            failed=view.failed_total > 0,
            completed_at=completed_timestamp,
            state=stage_status,
        )
    except Exception:  # pragma: no cover - defensive logging
        log.warning("NOTE_STYLE_MANIFEST_STAGE_UPDATE_FAILED sid=%s", sid, exc_info=True)

    try:
        record_stage(
            sid,
            "note_style",
            status=stage_status,
            counts=counts,
            empty_ok=empty_ok,
            metrics=metrics,
            results=results_counts,
            runs_root=runs_root_path,
        )
    except Exception:  # pragma: no cover - defensive logging
        log.warning("NOTE_STYLE_STAGE_RECORD_FAILED sid=%s", sid, exc_info=True)

    try:
        runflow_barriers_refresh(sid)
    except Exception:  # pragma: no cover - defensive logging
        log.warning(
            "NOTE_STYLE_BARRIERS_REFRESH_FAILED sid=%s", sid, exc_info=True
        )


def complete_note_style_result(
    sid: str,
    account_id: str,
    *,
    runs_root: Path | str | None = None,
    paths: NoteStylePaths | None = None,
    account_paths: NoteStyleAccountPaths | None = None,
    completed_at: str | None = None,
    update_stage: bool = True,
) -> tuple[Mapping[str, Any] | None, dict[str, int], int, bool]:
    """Mark ``account_id`` as completed and refresh note_style stage data."""

    runs_root_path = _resolve_runs_root(runs_root)
    ensure_note_style_section(sid, runs_root=runs_root_path)

    resolved_paths = paths or ensure_note_style_paths(runs_root_path, sid, create=True)
    resolved_account_paths = account_paths or ensure_note_style_account_paths(
        resolved_paths, account_id, create=True
    )

    writer = NoteStyleIndexWriter(sid=sid, paths=resolved_paths)
    updated_entry, totals, skipped_count, analysis_valid = writer.mark_completed(
        account_id,
        pack_path=resolved_account_paths.pack_file,
        result_path=resolved_account_paths.result_file,
        completed_at=completed_at,
    )

    if update_stage:
        if analysis_valid:
            _refresh_after_index_update(
                sid=sid,
                account_id=account_id,
                runs_root_path=runs_root_path,
                paths=resolved_paths,
                updated_entry=updated_entry,
                totals=totals,
                skipped_count=skipped_count,
                completed_at_override=completed_at,
            )
        else:
            log.warning(
                "NOTE_STYLE_REFRESH_DEFERRED sid=%s account_id=%s reason=%s",
                sid,
                account_id,
                "analysis_missing",
            )

    return updated_entry, totals, skipped_count, analysis_valid


def store_note_style_result(
    sid: str,
    account_id: str,
    payload: Mapping[str, Any],
    *,
    runs_root: Path | str | None = None,
    completed_at: str | None = None,
    update_index: bool = True,
) -> Path:
    """Persist the model ``payload`` for ``account_id`` and update the index."""

    runs_root_path = _resolve_runs_root(runs_root)
    ensure_note_style_section(sid, runs_root=runs_root_path)
    paths = ensure_note_style_paths(runs_root_path, sid, create=True)
    account_paths = ensure_note_style_account_paths(paths, account_id, create=True)

    normalized_payload = _normalize_result_payload(payload)

    note_metrics_payload = _resolve_note_metrics(
        normalized_payload,
        pack_path=account_paths.pack_file,
    )

    _apply_result_enhancements(
        normalized_payload,
        note_metrics=note_metrics_payload,
        pack_path=account_paths.pack_file,
    )

    note_text_value = _load_pack_note_text(account_paths.pack_file)
    if isinstance(note_text_value, str):
        normalized_note = note_text_value.strip()
        if normalized_note:
            note_hash = hashlib.sha256(
                normalized_note.encode("utf-8", "ignore")
            ).hexdigest()
            normalized_payload["note_hash"] = note_hash

    evaluated_at_value = normalized_payload.get("evaluated_at")
    if not isinstance(evaluated_at_value, str) or not evaluated_at_value.strip():
        normalized_completed_at = _normalize_timestamp_field(completed_at)
        normalized_payload["evaluated_at"] = (
            normalized_completed_at if normalized_completed_at else _now_iso()
        )

    _guard_result_analysis(
        sid=sid,
        account_id=account_id,
        paths=paths,
        payload=normalized_payload,
    )

    bytes_written = _atomic_write_jsonl(account_paths.result_file, normalized_payload)
    _validate_result_payload(
        sid=sid,
        account_id=account_id,
        paths=paths,
        account_paths=account_paths,
        payload=normalized_payload,
    )
    result_relative = _relative_to_base(account_paths.result_file, paths.base)
    log.info(
        "NOTE_STYLE_PARSED sid=%s account_id=%s path=%s bytes=%d",
        sid,
        account_id,
        result_relative,
        bytes_written,
    )

    analysis_payload = normalized_payload.get("analysis")
    tone_value: Any | None = None
    confidence_value: Any | None = None
    risk_flags_payload: list[Any] | None = None
    if isinstance(analysis_payload, Mapping):
        tone_value = analysis_payload.get("tone")
        confidence_value = analysis_payload.get("confidence")
        candidate_flags = analysis_payload.get("risk_flags")
        if isinstance(candidate_flags, Sequence) and not isinstance(
            candidate_flags, (str, bytes, bytearray)
        ):
            risk_flags_payload = list(candidate_flags)
        elif candidate_flags is None:
            risk_flags_payload = []

    log_structured_event(
        "NOTE_STYLE_PARSED",
        logger=log,
        sid=sid,
        account_id=account_id,
        result_path=result_relative,
        note_metrics=note_metrics_payload,
        tone=tone_value,
        confidence=confidence_value,
        risk_flags=risk_flags_payload,
    )

    if update_index:
        complete_note_style_result(
            sid,
            account_id,
            runs_root=runs_root_path,
            paths=paths,
            account_paths=account_paths,
            completed_at=completed_at,
        )

    return account_paths.result_file


def record_note_style_failure(
    sid: str,
    account_id: str,
    *,
    runs_root: Path | str | None = None,
    error: str | None = None,
) -> Path:
    """Record a failed note_style model attempt for ``account_id``."""

    runs_root_path = _resolve_runs_root(runs_root)
    ensure_note_style_section(sid, runs_root=runs_root_path)
    paths = ensure_note_style_paths(runs_root_path, sid, create=True)
    account_paths = ensure_note_style_account_paths(paths, account_id, create=True)

    writer = NoteStyleIndexWriter(sid=sid, paths=paths)
    updated_entry, totals, skipped_count = writer.mark_failed(
        account_id,
        pack_path=account_paths.pack_file,
        error=error,
    )

    log.info(
        "NOTE_STYLE_RESULT_FAILED sid=%s acc=%s error=%s",
        sid,
        account_id,
        error or "",
    )
    log_structured_event(
        "NOTE_STYLE_RESULT_FAILED",
        logger=log,
        sid=sid,
        account_id=account_id,
        error=error or "",
    )

    _refresh_after_index_update(
        sid=sid,
        account_id=account_id,
        runs_root_path=runs_root_path,
        paths=paths,
        updated_entry=updated_entry,
        totals=totals,
        skipped_count=skipped_count,
    )

    return account_paths.result_file


__all__ = [
    "NoteStyleIndexWriter",
    "complete_note_style_result",
    "record_note_style_failure",
    "store_note_style_result",
]
