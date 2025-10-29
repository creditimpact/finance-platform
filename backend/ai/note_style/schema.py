"""Schema definitions and validation helpers for note_style results."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from jsonschema import Draft7Validator


NOTE_STYLE_ANALYSIS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": [
        "tone",
        "context_hints",
        "emphasis",
        "confidence",
        "risk_flags",
    ],
    "additionalProperties": False,
    "properties": {
        "tone": {"type": "string", "minLength": 1},
        "context_hints": {
            "type": "object",
            "required": ["timeframe", "topic", "entities"],
            "additionalProperties": False,
            "properties": {
                "timeframe": {
                    "type": "object",
                    "required": ["month", "relative"],
                    "additionalProperties": False,
                    "properties": {
                        "month": {"type": ["string", "null"], "minLength": 1},
                        "relative": {"type": ["string", "null"], "minLength": 1},
                    },
                },
                "topic": {"type": "string", "minLength": 1},
                "entities": {
                    "type": "object",
                    "required": ["creditor", "amount"],
                    "additionalProperties": False,
                    "properties": {
                        "creditor": {"type": ["string", "null"], "minLength": 1},
                        "amount": {"type": ["number", "null"]},
                    },
                },
            },
        },
        "emphasis": {
            "type": "array",
            "items": {"type": "string", "minLength": 1},
        },
        "confidence": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
        },
        "risk_flags": {
            "type": "array",
            "items": {"type": "string", "minLength": 1},
        },
    },
}

# Exported for tool-calling parameter definitions.
NOTE_STYLE_TOOL_PARAMETERS_SCHEMA: dict[str, Any] = NOTE_STYLE_ANALYSIS_SCHEMA

_ANALYSIS_VALIDATOR = Draft7Validator(NOTE_STYLE_ANALYSIS_SCHEMA)


def validate_note_style_analysis(payload: Mapping[str, Any] | None) -> tuple[bool, list[str]]:
    """Validate ``payload`` against the strict note_style analysis schema."""

    if not isinstance(payload, Mapping):
        return False, ["payload_not_mapping"]

    errors = sorted(_ANALYSIS_VALIDATOR.iter_errors(payload), key=_error_sort_key)
    messages = [error.message for error in errors]
    return (not messages), messages


def _error_sort_key(error: Any) -> tuple[Sequence[Any], str]:
    path = tuple(error.path) if isinstance(error.path, Sequence) else ()
    return path, error.message


__all__ = [
    "NOTE_STYLE_ANALYSIS_SCHEMA",
    "NOTE_STYLE_TOOL_PARAMETERS_SCHEMA",
    "validate_note_style_analysis",
]
