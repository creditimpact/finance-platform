import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from logic.explanations_normalizer import sanitize, extract_structured, client


class DummyResponse:
    def __init__(self, payload: dict):
        text = json.dumps(payload)
        self.output = [type("o", (), {"content": [type("c", (), {"text": text})]})]


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

    def fake_create(*args, **kwargs):
        return DummyResponse(payload)

    monkeypatch.setattr(client.responses, "create", fake_create)
    result = extract_structured(safe_text, account_ctx)
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

