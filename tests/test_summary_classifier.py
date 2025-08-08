from logic.summary_classifier import classify_client_summary
from tests.helpers.fake_ai_client import FakeAIClient


def test_heuristic_identity_theft():
    summary = {
        "account_id": "1",
        "facts_summary": "This account is not mine and appears to be identity theft.",
        "claimed_errors": [],
    }
    res = classify_client_summary(summary, ai_client=FakeAIClient())
    assert res["category"] in {"identity_theft", "not_mine"}
    assert "FCRA" in res["legal_tag"]


def test_goodwill_mapping():
    summary = {
        "account_id": "2",
        "facts_summary": "I was late due to hardship and request goodwill.",
        "dispute_type": "goodwill",
        "claimed_errors": [],
    }
    res = classify_client_summary(summary, ai_client=FakeAIClient())
    assert res["category"] == "goodwill"
    assert res["dispute_approach"] == "goodwill_adjustment"
