from __future__ import annotations

import json

import pytest

from backend.ai.note_style.parse import (
    NoteStyleParseError,
    parse_note_style_response_payload,
)


def _make_response(payload: dict[str, object]) -> dict[str, object]:
    return {
        "mode": "content",
        "content_json": payload,
        "tool_json": None,
        "json": payload,
        "raw": None,
        "openai": None,
        "raw_content": json.dumps(payload),
        "raw_tool_arguments": None,
    }


def _make_tool_response(payload: dict[str, object]) -> dict[str, object]:
    return {
        "mode": "tool",
        "content_json": None,
        "tool_json": payload,
        "json": payload,
        "raw": None,
        "openai": None,
        "raw_content": None,
        "raw_tool_arguments": json.dumps(payload),
    }


def _analysis_payload() -> dict[str, object]:
    return {
        "tone": "formal",
        "context_hints": {
            "timeframe": {"month": "March", "relative": "last month"},
            "topic": "payment plan",
            "entities": {"creditor": "Example Bank", "amount": 125.5},
        },
        "emphasis": ["focus on empathy"],
        "confidence": 0.92,
        "risk_flags": ["compliance_check"],
    }


def _structured_payload() -> dict[str, object]:
    return {"note": "Example generated note", "analysis": _analysis_payload()}


def test_parse_pure_json_response() -> None:
    payload = _structured_payload()
    parsed = parse_note_style_response_payload(_make_response(payload))

    assert parsed.analysis == payload["analysis"]
    assert parsed.payload == payload
    assert parsed.source == "message.content:json"


def test_parse_fenced_json_response() -> None:
    payload = _structured_payload()
    parsed = parse_note_style_response_payload(_make_response(payload))

    assert parsed.analysis == payload["analysis"]
    assert parsed.payload == payload
    assert parsed.source == "message.content:json"


def test_parse_text_with_inline_json_response() -> None:
    payload = _structured_payload()
    parsed = parse_note_style_response_payload(_make_response(payload))

    assert parsed.analysis == payload["analysis"]
    assert parsed.source == "message.content:json"


def test_parse_malformed_json_raises() -> None:
    malformed = {"note": "", "analysis": {"tone": "missing required fields"}}

    with pytest.raises(NoteStyleParseError) as exc_info:
        parse_note_style_response_payload(_make_response(malformed))

    assert exc_info.value.code in {"invalid_schema", "schema_validation_failed", "invalid_json"}


def test_parse_uses_tool_call_arguments_when_content_missing() -> None:
    payload = _structured_payload()
    response = _make_tool_response(payload)

    parsed = parse_note_style_response_payload(response)

    assert parsed.analysis == payload["analysis"]
    assert parsed.payload == payload
    assert parsed.source == "message.tool_calls[0].function.arguments:json"


def test_parse_tool_call_mapping_arguments() -> None:
    payload = _structured_payload()
    response = _make_tool_response(payload)

    parsed = parse_note_style_response_payload(response)

    assert parsed.analysis == payload["analysis"]
    assert parsed.payload == payload
    assert parsed.source == "message.tool_calls[0].function.arguments:json"
