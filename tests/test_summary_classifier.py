from backend.core.logic.strategy.summary_classifier import (
    classify_client_summary,
    classify_client_summaries,
    invalidate_summary_cache,
)
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


def test_cache_and_invalidation():
    ai = FakeAIClient()
    ai.add_response('{"category": "not_mine"}')
    summary = {"account_id": "1", "facts_summary": "I dispute", "claimed_errors": []}
    res1 = classify_client_summary(
        summary, ai_client=ai, session_id="sess", account_id="1"
    )
    assert len(ai.chat_payloads) == 1
    res2 = classify_client_summary(
        summary, ai_client=ai, session_id="sess", account_id="1"
    )
    assert res1 == res2
    assert len(ai.chat_payloads) == 1
    invalidate_summary_cache("sess", "1")
    ai.add_response('{"category": "goodwill"}')
    classify_client_summary(summary, ai_client=ai, session_id="sess", account_id="1")
    assert len(ai.chat_payloads) == 2


def test_batch_classification():
    ai = FakeAIClient()
    ai.add_response(
        '{"1": {"category": "not_mine"}, "2": {"category": "goodwill"}}'
    )
    summaries = [
        {"account_id": "1", "facts_summary": "not mine", "claimed_errors": []},
        {"account_id": "2", "facts_summary": "goodwill", "claimed_errors": []},
    ]
    res = classify_client_summaries(summaries, ai_client=ai)
    assert res["1"]["category"] == "not_mine"
    assert res["2"]["category"] == "goodwill"
