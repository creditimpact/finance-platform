from __future__ import annotations

from backend.ai.note_style.schema import validate_note_style_analysis


def _valid_payload() -> dict[str, object]:
    return {
        "tone": "Empathetic",
        "context_hints": {
            "timeframe": {"month": "April", "relative": "last month"},
            "topic": "Payment plan",
            "entities": {"creditor": "Example Bank", "amount": 125.0},
        },
        "emphasis": ["support", "resolution"],
        "confidence": 0.92,
        "risk_flags": ["follow_up"],
    }


def test_schema_accepts_valid_payload() -> None:
    payload = _valid_payload()

    valid, errors = validate_note_style_analysis(payload)

    assert valid is True
    assert errors == []


def test_schema_rejects_missing_required_field() -> None:
    payload = _valid_payload()
    payload.pop("tone")

    valid, errors = validate_note_style_analysis(payload)

    assert valid is False
    assert any("tone" in message for message in errors)


def test_schema_rejects_invalid_confidence() -> None:
    payload = _valid_payload()
    payload["confidence"] = 5

    valid, errors = validate_note_style_analysis(payload)

    assert valid is False
    assert any(
        "confidence" in message.lower() or "maximum" in message.lower()
        for message in errors
    )
