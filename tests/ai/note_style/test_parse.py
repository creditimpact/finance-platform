from __future__ import annotations

import json

import pytest

from backend.ai.note_style.parse import (
    NoteStyleParseError,
    parse_note_style_response_text,
)


def _analysis_payload() -> dict[str, object]:
    return {
        "tone": "Supportive",
        "context_hints": {
            "timeframe": {"month": "June", "relative": "this month"},
            "topic": "Account assistance",
            "entities": {"creditor": "Example", "amount": 75.0},
        },
        "emphasis": ["empathetic"],
        "confidence": 0.87,
        "risk_flags": ["follow_up"],
    }


def test_parse_relaxed_python_literal_response() -> None:
    analysis = _analysis_payload()
    payload = {"analysis": analysis}
    relaxed_content = str(payload)

    parsed = parse_note_style_response_text(relaxed_content)

    assert parsed.analysis == analysis
    assert parsed.payload == payload
    assert parsed.source.endswith(":relaxed")


def test_parse_rejects_non_text_content() -> None:
    with pytest.raises(NoteStyleParseError) as exc_info:
        parse_note_style_response_text(123)  # type: ignore[arg-type]

    assert exc_info.value.code == "invalid_content_type"


def test_parse_requires_valid_schema() -> None:
    invalid_payload = json.dumps({"tone": "missing context"})

    with pytest.raises(NoteStyleParseError) as exc_info:
        parse_note_style_response_text(invalid_payload)

    assert exc_info.value.code == "schema_validation_failed"
