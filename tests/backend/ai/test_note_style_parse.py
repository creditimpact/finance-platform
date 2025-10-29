from __future__ import annotations

import json

import pytest

from backend.ai.note_style.parse import (
    NoteStyleParseError,
    parse_note_style_response_payload,
)


def _make_response(content: str) -> dict[str, object]:
    return {"choices": [{"message": {"content": content}}]}


def _make_tool_response(arguments: object, *, content: object | None = "") -> dict[str, object]:
    return {
        "choices": [
            {
                "message": {
                    "content": content,
                    "tool_calls": [
                        {"function": {"name": "submit_note_style_analysis", "arguments": arguments}}
                    ],
                }
            }
        ]
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


def test_parse_pure_json_response() -> None:
    analysis = _analysis_payload()
    content = json.dumps(analysis)
    parsed = parse_note_style_response_payload(_make_response(content))

    assert parsed.analysis == analysis
    assert parsed.payload == analysis
    assert parsed.source == "message.content:raw"


def test_parse_fenced_json_response() -> None:
    analysis = _analysis_payload()
    payload = {"analysis": analysis}
    content = f"Here you go:\n```json\n{json.dumps(payload)}\n```\nThanks!"

    parsed = parse_note_style_response_payload(_make_response(content))

    assert parsed.analysis == analysis
    assert parsed.payload == payload
    assert parsed.source.startswith("message.content:fenced#")


def test_parse_text_with_inline_json_response() -> None:
    analysis = _analysis_payload()
    payload = {"analysis": analysis}
    content = f"Summary: {json.dumps(payload)}. Let me know if you need anything else."

    parsed = parse_note_style_response_payload(_make_response(content))

    assert parsed.analysis == analysis
    assert parsed.source.startswith("message.content:")


def test_parse_malformed_json_raises() -> None:
    content = "Sure, here you go: {tone: 'missing quotes'}"

    with pytest.raises(NoteStyleParseError) as exc_info:
        parse_note_style_response_payload(_make_response(content))

    assert exc_info.value.code in {"invalid_json", "schema_validation_failed"}


def test_parse_uses_tool_call_arguments_when_content_missing() -> None:
    analysis = _analysis_payload()
    payload = {"analysis": analysis}
    response = _make_tool_response(json.dumps(payload), content=None)

    parsed = parse_note_style_response_payload(response)

    assert parsed.analysis == analysis
    assert parsed.payload == payload
    assert parsed.source.startswith("message.tool_calls[0].function.arguments:raw")


def test_parse_tool_call_mapping_arguments() -> None:
    analysis = _analysis_payload()
    payload = {"analysis": analysis}
    response = _make_tool_response(payload, content="")

    parsed = parse_note_style_response_payload(response)

    assert parsed.analysis == analysis
    assert parsed.payload == payload
    assert parsed.source.startswith("message.tool_calls[0].function.arguments:raw")
