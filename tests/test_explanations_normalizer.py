import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from backend.core.logic.letters.explanations_normalizer import (
    extract_structured,
    sanitize,
)
from tests.helpers.fake_ai_client import FakeAIClient


def test_sanitize_removes_unsafe_content():
    messy = "I <b>damn</b> paid 🤑 call me at 123-456-7890 or john@example.com"
    cleaned = sanitize(messy)
    assert cleaned == "I [REDACTED] paid call me at [REDACTED] or [REDACTED]"


def test_extract_structured_paraphrased_no_quotes(monkeypatch):
    safe_text = "Late payment due to bank error"
    account_ctx = {"account_id": "123", "dispute_type": "late_payment"}
    payload = {
        "account_id": "123",
        "dispute_type": "late_payment",
        "facts_summary": "Payment reported late though customer states bank mistake.",
        "claimed_errors": ["reported late despite timely payment"],
        "dates": {"incident": "2024-01-02"},
        "evidence": ["bank_statement"],
        "risk_flags": {"possible_identity_theft": False},
    }

    fake = FakeAIClient()
    fake.add_response(json.dumps(payload))
    result = extract_structured(safe_text, account_ctx, ai_client=fake)
    assert "Late payment due to bank error" not in result["facts_summary"]
    assert result["facts_summary"] == payload["facts_summary"]
    assert set(result.keys()) == {
        "account_id",
        "dispute_type",
        "facts_summary",
        "claimed_errors",
        "dates",
        "evidence",
        "risk_flags",
    }
    assert isinstance(result["claimed_errors"], list)
    assert isinstance(result["dates"], dict)


def test_extract_structured_pii_flag(monkeypatch):
    safe_text = sanitize("My SSN is 123-45-6789")
    account_ctx = {"account_id": "456", "dispute_type": "not_mine"}
    payload = {
        "account_id": "456",
        "dispute_type": "not_mine",
        "facts_summary": "Account reported in error.",
        "claimed_errors": [],
        "dates": {},
        "evidence": [],
        "risk_flags": {"contains_pii": True},
    }

    fake = FakeAIClient()
    fake.add_response(json.dumps(payload))
    result = extract_structured(safe_text, account_ctx, ai_client=fake)
    assert result["risk_flags"].get("contains_pii") is True


def test_extract_structured_filters_personal_emotional(monkeypatch):
    raw = (
        "I'm so sorry, I forgot to pay because my dog died and I was overwhelmed."
        " Please forgive me."
    )
    safe_text = sanitize(raw)
    account_ctx = {"account_id": "789", "dispute_type": "late_payment"}
    payload = {
        "account_id": "789",
        "dispute_type": "late_payment",
        "facts_summary": "Payment missed during personal hardship.",
        "claimed_errors": [],
        "dates": {},
        "evidence": [],
        "risk_flags": {},
    }

    fake = FakeAIClient()
    fake.add_response(json.dumps(payload))
    result = extract_structured(safe_text, account_ctx, ai_client=fake)
    import re

    summary = result["facts_summary"].lower()
    for term in ["sorry", "forgive", "dog"]:
        assert term not in summary
    assert re.search(r"\bi\b", summary) is None
