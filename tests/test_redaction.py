from importlib import reload

import backend.config as base_config
from backend.core.logic.report_analysis import redaction


def test_redaction_removes_pii(monkeypatch):
    monkeypatch.setenv("AI_REDACT_STRATEGY", "hash_last4")
    reload(base_config)
    reload(redaction)
    acct = {
        "account_number": "1234567890",
        "account_number_last4": "6789",
        "name": "John Doe",
        "ssn": "123-45-6789",
        "address": "123 Main St",
        "normalized_name": "bank a",
        "account_status": "Open",
        "balance_owed": 100,
    }
    redacted = redaction.redact_account_for_ai(acct)
    assert "account_number" not in redacted
    assert "name" not in redacted
    assert "ssn" not in redacted
    assert redacted["account_number_last4"] != "6789"
    assert redacted["normalized_name"] == "bank a"
    assert redacted["field_presence_map"]["account_status"] is True
    assert redacted["field_presence_map"]["payment_status"] is False
