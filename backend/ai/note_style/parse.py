"""Robust parsing utilities for note_style model responses."""

from __future__ import annotations

import ast
import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, MutableMapping, Sequence

from backend.ai.note_style.schema import validate_note_style_analysis


log = logging.getLogger(__name__)

_FENCED_BLOCK_PATTERN = re.compile(r"```(?:json|JSON)?\s*(?P<body>.+?)```", re.DOTALL)

_REQUIRED_ANALYSIS_KEYS = ("tone", "context_hints", "emphasis", "confidence", "risk_flags")
_REQUIRED_CONTEXT_KEYS = ("timeframe", "topic", "entities")
_REQUIRED_TIMEFRAME_KEYS = ("month", "relative")
_REQUIRED_ENTITY_KEYS = ("creditor", "amount")


def _normalize_nullable_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _normalize_required_str(value: Any) -> str:
    return str(value).strip()


def _normalize_timeframe_month(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not value.is_integer():
            return str(value)
        month_int = int(value)
        if 1 <= month_int <= 12:
            return f"{month_int:02d}"
        return str(month_int)
    text = str(value).strip()
    return text or None


def _normalize_amount(value: Any) -> float | None:
    if value in (None, ""):
        return None
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    cleaned = text.replace(",", "")
    if cleaned.startswith("$"):
        cleaned = cleaned[1:]
    try:
        return float(cleaned)
    except ValueError:
        return value


def _normalize_string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        entry = value.strip()
        return [entry] if entry else []
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        normalized: list[str] = []
        for item in value:
            if item is None:
                continue
            text = str(item).strip()
            if text:
                normalized.append(text)
        return normalized
    return []


def _normalize_confidence_value(value: Any) -> Any:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return value
        try:
            return float(text)
        except ValueError:
            return value
    return value


def _normalize_note_style_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("note_style payload must be a JSON object")

    # Unwrap accidental envelope
    if "analysis" in payload and isinstance(payload["analysis"], dict):
        payload = dict(payload["analysis"])
    else:
        payload = dict(payload)

    # Drop accidental 'note' key if present
    payload.pop("note", None)

    tone_value = payload.get("tone")
    if not isinstance(tone_value, str) or not tone_value.strip():
        payload["tone"] = "unspecified"

    payload.setdefault("context_hints", {})

    if "emphasis" not in payload or payload["emphasis"] is None:
        payload["emphasis"] = []

    if "confidence" not in payload or payload["confidence"] is None:
        payload["confidence"] = 0.0

    if "risk_flags" not in payload or payload["risk_flags"] is None:
        payload["risk_flags"] = []

    ch = payload["context_hints"]
    if not isinstance(ch, dict):
        ch = {}
    payload["context_hints"] = ch
    ch.setdefault("timeframe", {})
    ch.setdefault("topic", "unspecified")
    ch.setdefault("entities", {})

    tf = ch["timeframe"]
    if not isinstance(tf, dict):
        tf = {}
    ch["timeframe"] = tf
    tf.setdefault("month", None)
    tf.setdefault("relative", None)

    topic_value = ch.get("topic")
    if not isinstance(topic_value, str) or not topic_value.strip():
        ch["topic"] = "unspecified"

    ent = ch["entities"]
    if not isinstance(ent, dict):
        ent = {}
    ch["entities"] = ent
    ent.setdefault("creditor", None)
    ent.setdefault("amount", None)

    # Coerce month number → string
    m = tf.get("month")
    if isinstance(m, (int, float)) and m == int(m):
        tf["month"] = f"{int(m):02d}"
    elif m is not None and not isinstance(m, str):
        tf["month"] = str(m)

    return payload


def _normalize_analysis_payload(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        return payload

    normalized: dict[str, Any] = {}

    if "tone" in payload:
        tone_value = payload.get("tone")
        if tone_value is not None:
            normalized["tone"] = _normalize_required_str(tone_value)

    context_value = payload.get("context_hints")
    if isinstance(context_value, Mapping):
        context_normalized: dict[str, Any] = {}

        timeframe_value = context_value.get("timeframe")
        if isinstance(timeframe_value, Mapping):
            timeframe_normalized: dict[str, Any] = {}
            if "month" in timeframe_value:
                timeframe_normalized["month"] = _normalize_timeframe_month(
                    timeframe_value.get("month")
                )
            if "relative" in timeframe_value:
                timeframe_normalized["relative"] = _normalize_nullable_str(
                    timeframe_value.get("relative")
                )
            context_normalized["timeframe"] = timeframe_normalized

        topic_value = context_value.get("topic")
        if topic_value is not None:
            context_normalized["topic"] = _normalize_required_str(topic_value)

        entities_value = context_value.get("entities")
        if isinstance(entities_value, Mapping):
            entities_normalized: dict[str, Any] = {}
            if "creditor" in entities_value:
                entities_normalized["creditor"] = _normalize_nullable_str(
                    entities_value.get("creditor")
                )
            if "amount" in entities_value:
                entities_normalized["amount"] = _normalize_amount(
                    entities_value.get("amount")
                )
            context_normalized["entities"] = entities_normalized

        if context_normalized:
            normalized["context_hints"] = context_normalized

    if "emphasis" in payload:
        normalized["emphasis"] = _normalize_string_list(payload.get("emphasis"))

    if "confidence" in payload:
        normalized["confidence"] = _normalize_confidence_value(payload.get("confidence"))

    if "risk_flags" in payload:
        normalized["risk_flags"] = _normalize_string_list(payload.get("risk_flags"))

    return normalized


@dataclass(frozen=True)
class NoteStyleParsedResponse:
    """Represents a successfully parsed note_style response."""

    payload: Mapping[str, Any]
    analysis: Mapping[str, Any]
    source: str


class NoteStyleParseError(ValueError):
    """Raised when a note_style model response cannot be parsed strictly."""

    def __init__(self, message: str, *, code: str, details: Mapping[str, Any] | None = None):
        super().__init__(f"{code}: {message}")
        self.code = code
        self.details = dict(details) if isinstance(details, Mapping) else {}


def _coerce_attr(payload: Any, name: str) -> Any:
    if hasattr(payload, name):
        return getattr(payload, name)
    if isinstance(payload, Mapping):
        return payload.get(name)
    return None


def _extract_tool_call_arguments(message: Any) -> str | None:
    tool_calls = _coerce_attr(message, "tool_calls")
    if not isinstance(tool_calls, Sequence) or not tool_calls:
        return None

    first_call = tool_calls[0]
    function_payload = _coerce_attr(first_call, "function")
    if function_payload is None:
        return None

    arguments = _coerce_attr(function_payload, "arguments")
    if isinstance(arguments, str):
        candidate = arguments.strip()
        return candidate or None

    if isinstance(arguments, Mapping) or (
        isinstance(arguments, Sequence) and not isinstance(arguments, (bytes, bytearray))
    ):
        try:
            serialized = json.dumps(arguments, ensure_ascii=False)
        except (TypeError, ValueError):
            return None
        candidate = serialized.strip()
        return candidate or None

    if arguments is not None:
        candidate = str(arguments).strip()
        return candidate or None

    return None


def _extract_response_content(response_payload: Any) -> tuple[str, str]:
    choices: Sequence[Any] | None = _coerce_attr(response_payload, "choices")
    if not isinstance(choices, Sequence) or not choices:
        raise NoteStyleParseError("Model response missing choices", code="missing_choices")

    first = choices[0]
    message = _coerce_attr(first, "message")
    if message is None:
        raise NoteStyleParseError("Model response missing message", code="missing_message")

    content = _coerce_attr(message, "content")
    if isinstance(content, str):
        text = content
    elif isinstance(content, Sequence) and not isinstance(content, (bytes, bytearray)):
        pieces: list[str] = []
        for chunk in content:
            if isinstance(chunk, str):
                pieces.append(chunk)
            elif isinstance(chunk, Mapping):
                text_piece = chunk.get("text")
                if isinstance(text_piece, str):
                    pieces.append(text_piece)
        text = "".join(pieces)
    elif content is None:
        text = ""
    else:
        text = str(content)

    normalized = text.strip()
    if normalized:
        return normalized, "message.content"

    tool_arguments = _extract_tool_call_arguments(message)
    if tool_arguments:
        return tool_arguments, "message.tool_calls[0].function.arguments"

    if content is None:
        raise NoteStyleParseError("Model response missing content", code="missing_content")

    raise NoteStyleParseError("Model response missing JSON content", code="empty_content")


def _extract_fenced_candidates(text: str) -> Iterable[str]:
    for match in _FENCED_BLOCK_PATTERN.finditer(text):
        body = match.group("body")
        if isinstance(body, str):
            stripped = body.strip()
            if stripped:
                yield stripped


def _extract_longest_object(text: str) -> str | None:
    start_stack: list[int] = []
    best_span: tuple[int, int] | None = None
    for index, char in enumerate(text):
        if char == "{":
            start_stack.append(index)
        elif char == "}" and start_stack:
            start = start_stack.pop()
            if not start_stack:
                span = (start, index + 1)
                if best_span is None or (span[1] - span[0]) > (best_span[1] - best_span[0]):
                    best_span = span
    if best_span is None:
        return None
    candidate = text[best_span[0] : best_span[1]].strip()
    return candidate or None


def _generate_candidates(text: str) -> list[tuple[str, str]]:
    trimmed = text.strip()
    raw_candidates: list[tuple[str, str]] = [("raw", trimmed)] if trimmed else []

    fenced = list(_extract_fenced_candidates(text))
    raw_candidates.extend((f"fenced#{index + 1}", value) for index, value in enumerate(fenced))

    longest = _extract_longest_object(text)
    if longest:
        raw_candidates.append(("object", longest))

    deduped: list[tuple[str, str]] = []
    seen_values: set[str] = set()
    for label, candidate in raw_candidates:
        if candidate in seen_values:
            continue
        seen_values.add(candidate)
        deduped.append((label, candidate))
    return deduped


def _schema_validate(payload: Mapping[str, Any]) -> tuple[bool, list[str]]:
    return validate_note_style_analysis(payload)


def _strict_parser_enabled() -> bool:
    value = os.getenv("NOTE_STYLE_PARSER_STRICT")
    if value is None:
        return True
    lowered = value.strip().lower()
    if lowered in {"", "1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    return True


def _gather_analysis_field_errors(analysis: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    missing_keys = [key for key in _REQUIRED_ANALYSIS_KEYS if key not in analysis]
    if missing_keys:
        errors.append(
            "analysis missing keys: " + ", ".join(sorted(missing_keys))
        )

    context = analysis.get("context_hints")
    if not isinstance(context, Mapping):
        errors.append("analysis.context_hints must be an object")
    else:
        context_missing = [key for key in _REQUIRED_CONTEXT_KEYS if key not in context]
        if context_missing:
            errors.append(
                "analysis.context_hints missing keys: "
                + ", ".join(sorted(context_missing))
            )

        timeframe = context.get("timeframe")
        if not isinstance(timeframe, Mapping):
            errors.append("analysis.context_hints.timeframe must be an object")
        else:
            timeframe_missing = [
                key for key in _REQUIRED_TIMEFRAME_KEYS if key not in timeframe
            ]
            if timeframe_missing:
                errors.append(
                    "analysis.context_hints.timeframe missing keys: "
                    + ", ".join(sorted(timeframe_missing))
                )

        entities = context.get("entities") if isinstance(context, Mapping) else None
        if not isinstance(entities, Mapping):
            errors.append("analysis.context_hints.entities must be an object")
        else:
            entity_missing = [
                key for key in _REQUIRED_ENTITY_KEYS if key not in entities
            ]
            if entity_missing:
                errors.append(
                    "analysis.context_hints.entities missing keys: "
                    + ", ".join(sorted(entity_missing))
                )

    return errors


def _ensure_note_analysis_structure(payload: Mapping[str, Any], *, source: str) -> None:
    if not isinstance(payload, Mapping):
        raise NoteStyleParseError(
            "note_style analysis payload must be an object",
            code="invalid_schema",
            details={"source": source},
        )

    errors = _gather_analysis_field_errors(payload)

    if errors:
        raise NoteStyleParseError(
            "Model response missing required analysis fields",
            code="invalid_schema",
            details={"source": source, "messages": errors},
        )


def _coerce_relaxed_value(value: Any) -> Any:
    if isinstance(value, dict):
        coerced: MutableMapping[str, Any] = {}
        for key, entry in value.items():
            coerced[str(key)] = _coerce_relaxed_value(entry)
        return coerced
    if isinstance(value, list):
        return [_coerce_relaxed_value(item) for item in value]
    if isinstance(value, tuple):
        return [_coerce_relaxed_value(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _attempt_relaxed_parse(text: str) -> Mapping[str, Any] | None:
    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return None
    coerced = _coerce_relaxed_value(parsed)
    if not isinstance(coerced, Mapping):
        return None
    try:
        normalized_text = json.dumps(coerced, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError):
        return None
    try:
        return json.loads(normalized_text)
    except json.JSONDecodeError:
        return None


def _summarize_candidate(label: str, candidate: str) -> str:
    preview = candidate.strip().replace("\n", " ")
    if len(preview) > 120:
        preview = preview[:117] + "..."
    return f"{label}: {preview}"


def parse_note_style_response_text(
    text: str,
    *,
    origin: str = "message.content",
    strict: bool | None = None,
) -> NoteStyleParsedResponse:
    """Parse ``text`` into a validated :class:`NoteStyleParsedResponse`."""

    if not isinstance(text, str):
        raise NoteStyleParseError("Response content must be text", code="invalid_content_type")

    strict_mode = _strict_parser_enabled() if strict is None else strict

    candidates = _generate_candidates(text)
    if not candidates:
        raise NoteStyleParseError("Model response missing JSON content", code="empty_content")

    attempted: list[Mapping[str, Any]] = []
    attempt_details: list[Mapping[str, Any]] = []

    for label, candidate in candidates:
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError as exc:
            attempt_details.append(
                {
                    "source": label,
                    "error": "json_decode_error",
                    "message": str(exc),
                    "preview": _summarize_candidate(label, candidate),
                }
            )
            continue
        if not isinstance(payload, Mapping):
            attempt_details.append(
                {
                    "source": label,
                    "error": "not_mapping",
                    "preview": _summarize_candidate(label, candidate),
                }
            )
            continue

        try:
            normalized_payload = _normalize_note_style_payload(dict(payload))
        except ValueError as exc:
            attempt_details.append(
                {
                    "source": label,
                    "error": "normalization_failed",
                    "message": str(exc),
                    "preview": _summarize_candidate(label, candidate),
                }
            )
            continue

        normalized_analysis = _normalize_analysis_payload(normalized_payload)
        valid, errors = _schema_validate(normalized_analysis)
        if valid:
            response = NoteStyleParsedResponse(
                payload=normalized_payload,
                analysis=normalized_analysis,
                source=f"{origin}:{label}",
            )
            if strict_mode:
                _ensure_note_analysis_structure(normalized_payload, source=response.source)
            return response

        attempted.append(normalized_payload)
        attempt_details.append(
            {
                "source": label,
                "error": "schema_validation_failed",
                "messages": errors,
                "preview": _summarize_candidate(label, candidate),
            }
        )

    if not strict_mode:
        for label, candidate in candidates:
            relaxed = _attempt_relaxed_parse(candidate)
            if relaxed is None:
                continue
            try:
                normalized_payload = _normalize_note_style_payload(dict(relaxed))
            except ValueError as exc:
                attempt_details.append(
                    {
                        "source": f"{label}:relaxed",
                        "error": "normalization_failed",
                        "message": str(exc),
                        "preview": _summarize_candidate(label, candidate),
                    }
                )
                continue

            normalized_analysis = _normalize_analysis_payload(normalized_payload)
            valid, errors = _schema_validate(normalized_analysis)
            if valid:
                response = NoteStyleParsedResponse(
                    payload=normalized_payload,
                    analysis=normalized_analysis,
                    source=f"{origin}:{label}:relaxed",
                )
                return response

            attempted.append(normalized_payload)
            attempt_details.append(
                {
                    "source": f"{label}:relaxed",
                    "error": "schema_validation_failed",
                    "messages": errors,
                    "preview": _summarize_candidate(label, candidate),
                }
            )

    error_code = "schema_validation_failed" if attempted else "invalid_json"
    raise NoteStyleParseError(
        "Model response missing valid note_style analysis JSON",
        code=error_code,
        details={"attempts": attempt_details[:5]},
    )

def _build_validated_response(
    payload: Mapping[str, Any],
    *,
    origin: str,
    source: str,
    strict: bool,
) -> NoteStyleParsedResponse:
    normalized_payload = _normalize_note_style_payload(dict(payload))

    if strict:
        _ensure_note_analysis_structure(normalized_payload, source=source)

    normalized_analysis = _normalize_analysis_payload(normalized_payload)
    valid, errors = _schema_validate(normalized_analysis)
    if not valid:
        raise NoteStyleParseError(
            "Model response missing valid note_style analysis JSON",
            code="schema_validation_failed",
            details={
                "source": source,
                "messages": errors,
            },
        )

    return NoteStyleParsedResponse(
        payload=normalized_payload, analysis=normalized_analysis, source=source
    )


def _parse_normalized_response_payload(
    response_payload: Mapping[str, Any],
    *,
    strict: bool,
) -> NoteStyleParsedResponse:
    mode = str(response_payload.get("mode") or "").strip()

    if mode == "tool":
        origin = "message.tool_calls[0].function.arguments"
        json_payload = response_payload.get("tool_json")
    else:
        origin = "message.content"
        expects_content_json = "content_json" in response_payload
        json_payload = response_payload.get("content_json")
        if expects_content_json:
            if json_payload is None:
                raise ValueError("note_style expects JSON object in content_json, got none")
            if not isinstance(json_payload, Mapping):
                raise ValueError("note_style expects JSON object in content_json, got non-object")

    if not isinstance(json_payload, Mapping):
        fallback = response_payload.get("json")
        if isinstance(fallback, Mapping) and (
            mode == "tool" or "content_json" not in response_payload
        ):
            json_payload = fallback

    if not isinstance(json_payload, Mapping) and mode != "tool":
        content_text = response_payload.get("content_text")
        if isinstance(content_text, str) and content_text.strip():
            return parse_note_style_response_text(
                content_text, origin=origin, strict=strict
            )
        if "content_text" in response_payload and content_text is None:
            raise ValueError(
                "note_style received explicit content_text=None without content_json"
            )

    if isinstance(json_payload, Mapping):
        return _build_validated_response(
            json_payload,
            origin=origin,
            source=f"{origin}:json",
            strict=strict,
        )

    # Fall back to attempting to parse the raw text if available.
    if mode == "tool":
        raw_candidate = response_payload.get("raw_tool_arguments")
    else:
        raw_candidate = response_payload.get("raw_content")

    if isinstance(raw_candidate, str) and raw_candidate.strip():
        return parse_note_style_response_text(raw_candidate, origin=origin, strict=strict)

    if raw_candidate is not None and mode == "tool":
        try:
            serialized = json.dumps(raw_candidate, ensure_ascii=False)
        except (TypeError, ValueError):
            pass
        else:
            if serialized.strip():
                return parse_note_style_response_text(
                    serialized, origin=origin, strict=strict
                )

    raise NoteStyleParseError("Model response missing JSON content", code="empty_content")


def parse_note_style_response_payload(response_payload: Any) -> NoteStyleParsedResponse:
    """Parse and validate the ``response_payload`` from the model."""

    strict_mode = _strict_parser_enabled()

    candidate_payload: Mapping[str, Any] | None = None
    candidate_source = "response_payload"
    if isinstance(response_payload, Mapping):
        for key, source in (
            ("json", "response_payload.json"),
            ("content_json", "response_payload.content_json"),
            ("tool_json", "response_payload.tool_json"),
        ):
            value = response_payload.get(key)
            if isinstance(value, Mapping) and ("analysis" in value or "note" in value):
                candidate_payload = value
                candidate_source = source
                break
        else:
            if "analysis" in response_payload or "note" in response_payload:
                candidate_payload = response_payload  # type: ignore[assignment]

    if strict_mode and candidate_payload is not None:
        normalized_candidate = _normalize_note_style_payload(dict(candidate_payload))
        _ensure_note_analysis_structure(normalized_candidate, source=candidate_source)

    if isinstance(response_payload, Mapping) and ("mode" in response_payload or "json" in response_payload):
        return _parse_normalized_response_payload(response_payload, strict=strict_mode)

    text, origin = _extract_response_content(response_payload)
    return parse_note_style_response_text(text, origin=origin, strict=strict_mode)


__all__ = [
    "NoteStyleParseError",
    "NoteStyleParsedResponse",
    "parse_note_style_response_payload",
    "parse_note_style_response_text",
]
