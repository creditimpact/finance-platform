import re

from backend.validation.redaction import (
    sanitize_validation_log_payload,
    sanitize_validation_payload,
)


def test_sanitize_validation_payload_masks_account_and_contact_info() -> None:
    payload = {
        "field": "account_number_display",
        "bureaus": {
            "transunion": {
                "raw": "1234567890",
                "normalized": {"display": "1234-567-890", "last4": "7890"},
            }
        },
        "context": {
            "contact_name": "John Doe",
            "contact_phone": "555-123-4567",
        },
    }

    sanitized = sanitize_validation_payload(payload)

    tu = sanitized["bureaus"]["transunion"]
    assert tu["raw"] == "******7890"
    display = tu["normalized"]["display"]
    assert display.count("*") >= 6
    assert re.sub(r"\D", "", display).endswith("7890")
    assert tu["normalized"]["last4"] == "7890"

    context = sanitized["context"]
    assert context["contact_name"] == "[REDACTED_NAME]"
    assert context["contact_phone"] == "[REDACTED_PHONE]"


def test_sanitize_validation_log_payload_masks_nested_keys() -> None:
    payload = {
        "account_number_display": "55551234",
        "metadata": {
            "agentName": "Jane Smith",
            "agent_phone": "555-987-6543",
        },
    }

    sanitized = sanitize_validation_log_payload(payload)

    assert sanitized["account_number_display"] == "****1234"
    assert sanitized["metadata"]["agentName"] == "[REDACTED_NAME]"
    assert sanitized["metadata"]["agent_phone"] == "[REDACTED_PHONE]"
