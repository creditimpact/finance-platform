import json

from backend.core.logic.strategy.summary_classifier import (
    classify_client_summary,
    classify_client_summaries,
    invalidate_summary_cache,
    reset_cache,
)
from tests.helpers.fake_ai_client import FakeAIClient


def test_heuristic_identity_theft():
    summary = {
        "account_id": "1",
        "facts_summary": "This account is not mine and appears to be identity theft.",
        "claimed_errors": [],
    }
    ai = FakeAIClient()
    res = classify_client_summary(summary, ai_client=ai)
    assert res["category"] in {"identity_theft", "not_mine"}
    assert "FCRA" in res["legal_tag"]
    assert len(ai.chat_payloads) == 0


def test_goodwill_mapping():
    summary = {
        "account_id": "2",
        "facts_summary": "I was late due to hardship and request goodwill.",
        "dispute_type": "goodwill",
        "claimed_errors": [],
    }
    ai = FakeAIClient()
    res = classify_client_summary(summary, ai_client=ai)
    assert res["category"] == "goodwill"
    assert res["dispute_approach"] == "goodwill_adjustment"
    assert len(ai.chat_payloads) == 0


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


def test_empty_summary_bypasses_ai():
    ai = FakeAIClient()
    summary = {"account_id": "3", "facts_summary": "", "claimed_errors": []}
    res = classify_client_summary(summary, ai_client=ai)
    assert res["category"] == "inaccurate_reporting"
    assert len(ai.chat_payloads) == 0


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


def test_batch_empty_summary_skips_ai():
    ai = FakeAIClient()
    ai.add_response('{"2": {"category": "goodwill"}}')
    summaries = [
        {"account_id": "1", "facts_summary": "   ", "claimed_errors": []},
        {"account_id": "2", "facts_summary": "goodwill", "claimed_errors": []},
    ]
    res = classify_client_summaries(summaries, ai_client=ai)
    assert res["1"]["category"] == "inaccurate_reporting"
    assert res["2"]["category"] == "goodwill"
    assert len(ai.chat_payloads) == 1


def test_batch_long_summary_cached_and_heuristic():
    reset_cache()
    long_text = ("x" * 5000) + " not mine"
    summary = {"account_id": "1", "facts_summary": long_text, "claimed_errors": []}
    ai = FakeAIClient()
    ai.add_response("{}")
    res1 = classify_client_summaries([summary], ai_client=ai, session_id="sess_long")
    assert res1["1"]["category"] == "not_mine"
    res2 = classify_client_summaries([summary], ai_client=ai, session_id="sess_long")
    assert res2["1"]["category"] == "not_mine"
    assert len(ai.chat_payloads) == 1


def test_batch_fallback_missing_items_preserves_map():
    ai = FakeAIClient()
    batch_resp = {str(i): {"category": "not_mine"} for i in range(1, 9)}
    ai.add_response(json.dumps(batch_resp))
    ai.add_response('{"category": "goodwill"}')
    ai.add_response('{"category": "identity_theft"}')
    summaries = [
        {"account_id": str(i), "facts_summary": "not mine", "claimed_errors": []}
        if i <= 8
        else {"account_id": str(i), "facts_summary": "mystery", "claimed_errors": []}
        for i in range(1, 11)
    ]
    res = classify_client_summaries(summaries, ai_client=ai)
    assert len(res) == 10
    for i in range(1, 9):
        assert res[str(i)]["category"] == "not_mine"
    assert res["9"]["category"] == "goodwill"
    assert res["10"]["category"] == "identity_theft"
    assert len(ai.chat_payloads) == 3
