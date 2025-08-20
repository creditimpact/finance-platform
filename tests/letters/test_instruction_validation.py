import pytest

from backend.core.letters.router import select_template
from backend.analytics.analytics_tracker import reset_counters


def test_missing_field_failure(monkeypatch):
    monkeypatch.setenv("LETTERS_ROUTER_PHASED", "1")
    reset_counters()
    decision = select_template("bureau_dispute", {}, phase="finalize")
    assert "bureau" in decision.missing_fields


def test_substance_failure(monkeypatch):
    monkeypatch.setenv("LETTERS_ROUTER_PHASED", "1")
    reset_counters()
    ctx = {
        "creditor_name": "Cred",
        "account_number_masked": "1234",
        "legal_safe_summary": "summary",
        "cra_last_result": "result",
        "days_since_cra_result": 30,
    }
    decision = select_template("mov", ctx, phase="finalize")
    assert "reinvestigation_request" in decision.missing_fields
    assert "method_of_verification" in decision.missing_fields
