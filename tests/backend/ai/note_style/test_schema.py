from backend.ai.note_style.schema import validate_note_style_analysis


def _build_valid_payload() -> dict[str, object]:
    return {
        "tone": "formal",
        "context_hints": {
            "timeframe": {"month": "2024-01", "relative": "last month"},
            "topic": "account dispute",
            "entities": {"creditor": "Acme Bank", "amount": 1234.56},
        },
        "emphasis": ["highlight billing error"],
        "confidence": 0.82,
        "risk_flags": ["legal_risk"],
    }


def test_validate_note_style_analysis_accepts_valid_payload() -> None:
    payload = _build_valid_payload()

    valid, errors = validate_note_style_analysis(payload)

    assert valid is True
    assert errors == []


def test_validate_note_style_analysis_rejects_missing_required_fields() -> None:
    payload = _build_valid_payload()
    payload.pop("tone")

    valid, errors = validate_note_style_analysis(payload)

    assert valid is False
    assert errors


def test_validate_note_style_analysis_rejects_invalid_types() -> None:
    payload = _build_valid_payload()
    payload["emphasis"] = "not-a-list"  # type: ignore[assignment]

    valid, errors = validate_note_style_analysis(payload)

    assert valid is False
    assert any("array" in message or "type" in message for message in errors)
