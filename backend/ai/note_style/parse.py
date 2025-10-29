"""Robust parsing utilities for note_style model responses."""

from __future__ import annotations

import ast
import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, MutableMapping, Sequence

from backend.ai.note_style.schema import validate_note_style_analysis


log = logging.getLogger(__name__)

_FENCED_BLOCK_PATTERN = re.compile(r"```(?:json|JSON)?\s*(?P<body>.+?)```", re.DOTALL)


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
) -> NoteStyleParsedResponse:
    """Parse ``text`` into a validated :class:`NoteStyleParsedResponse`."""

    if not isinstance(text, str):
        raise NoteStyleParseError("Response content must be text", code="invalid_content_type")

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

        analysis_payload, path = _resolve_analysis_payload(payload)
        valid, errors = _schema_validate(analysis_payload)
        if valid:
            return NoteStyleParsedResponse(
                payload=payload,
                analysis=analysis_payload,
                source=f"{origin}:{label}",
            )

        attempted.append(payload)
        attempt_details.append(
            {
                "source": label,
                "error": "schema_validation_failed",
                "messages": errors,
                "preview": _summarize_candidate(label, candidate),
                "path": path,
            }
        )

    for label, candidate in candidates:
        relaxed = _attempt_relaxed_parse(candidate)
        if relaxed is None:
            continue
        analysis_payload, path = _resolve_analysis_payload(relaxed)
        valid, errors = _schema_validate(analysis_payload)
        if valid:
            return NoteStyleParsedResponse(
                payload=relaxed,
                analysis=analysis_payload,
                source=f"{origin}:{label}:relaxed",
            )

        attempted.append(relaxed)
        attempt_details.append(
            {
                "source": f"{label}:relaxed",
                "error": "schema_validation_failed",
                "messages": errors,
                "preview": _summarize_candidate(label, candidate),
                "path": path,
            }
        )

    error_code = "schema_validation_failed" if attempted else "invalid_json"
    raise NoteStyleParseError(
        "Model response missing valid note_style analysis JSON",
        code=error_code,
        details={"attempts": attempt_details[:5]},
    )


def _resolve_analysis_payload(payload: Mapping[str, Any]) -> tuple[Mapping[str, Any], str]:
    if isinstance(payload.get("analysis"), Mapping):
        analysis = payload.get("analysis")  # type: ignore[assignment]
        return analysis, "analysis"
    return payload, "root"


def _build_validated_response(
    payload: Mapping[str, Any],
    *,
    origin: str,
    source: str,
) -> NoteStyleParsedResponse:
    analysis_payload, path = _resolve_analysis_payload(payload)
    valid, errors = _schema_validate(analysis_payload)
    if not valid:
        raise NoteStyleParseError(
            "Model response missing valid note_style analysis JSON",
            code="schema_validation_failed",
            details={
                "source": source,
                "messages": errors,
                "path": path,
            },
        )

    return NoteStyleParsedResponse(payload=payload, analysis=analysis_payload, source=source)


def _parse_normalized_response_payload(
    response_payload: Mapping[str, Any],
) -> NoteStyleParsedResponse:
    mode = str(response_payload.get("mode") or "").strip()

    if mode == "tool":
        origin = "message.tool_calls[0].function.arguments"
        json_payload = response_payload.get("tool_json")
    else:
        origin = "message.content"
        json_payload = response_payload.get("content_json")

    if not isinstance(json_payload, Mapping):
        fallback = response_payload.get("json")
        if isinstance(fallback, Mapping):
            json_payload = fallback

    if isinstance(json_payload, Mapping):
        return _build_validated_response(json_payload, origin=origin, source=f"{origin}:json")

    # Fall back to attempting to parse the raw text if available.
    if mode == "tool":
        raw_candidate = response_payload.get("raw_tool_arguments")
    else:
        raw_candidate = response_payload.get("raw_content")

    if isinstance(raw_candidate, str) and raw_candidate.strip():
        return parse_note_style_response_text(raw_candidate, origin=origin)

    if raw_candidate is not None and mode == "tool":
        try:
            serialized = json.dumps(raw_candidate, ensure_ascii=False)
        except (TypeError, ValueError):
            pass
        else:
            if serialized.strip():
                return parse_note_style_response_text(serialized, origin=origin)

    raise NoteStyleParseError("Model response missing JSON content", code="empty_content")


def parse_note_style_response_payload(response_payload: Any) -> NoteStyleParsedResponse:
    """Parse and validate the ``response_payload`` from the model."""

    if isinstance(response_payload, Mapping) and ("mode" in response_payload or "json" in response_payload):
        try:
            return _parse_normalized_response_payload(response_payload)
        except NoteStyleParseError:
            raise

    text, origin = _extract_response_content(response_payload)
    return parse_note_style_response_text(text, origin=origin)


__all__ = [
    "NoteStyleParseError",
    "NoteStyleParsedResponse",
    "parse_note_style_response_payload",
    "parse_note_style_response_text",
]
