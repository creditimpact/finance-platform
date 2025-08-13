import importlib
import time

from backend.core.logic.compliance.rules_loader import recompute_rules_version
from backend.core.logic.letters.generate_custom_letters import (
    call_gpt_for_custom_letter,
)
from backend.core.logic.letters.goodwill_preparation import prepare_account_summaries
from backend.core.logic.strategy.summary_classifier import (
    ClassificationRecord,
    summary_hash,
)
from backend.core.orchestrators import classify_client_responses
from tests.helpers.fake_ai_client import FakeAIClient


def test_classify_client_responses_writes_metadata(monkeypatch):
    store: dict[str, dict] = {}

    def fake_update(session_id: str, **kwargs):
        data = store.setdefault(session_id, {})
        for k, v in kwargs.items():
            if isinstance(v, dict) and isinstance(data.get(k), dict):
                data[k].update(v)
            else:
                data[k] = v
        return data

    def fake_get(session_id: str):
        return store.get(session_id)

    monkeypatch.setattr("backend.api.session_manager.update_session", fake_update)
    monkeypatch.setattr("backend.api.session_manager.get_session", fake_get)

    structured = {
        "1": {"account_id": "1", "facts_summary": "hello", "claimed_errors": []}
    }
    raw = {"1": "hello"}
    client_info = {"session_id": "sess1", "state": "CA"}

    ai = FakeAIClient()
    ai.add_response('{"1": {"category": "not_mine"}}')

    classify_client_responses(
        structured, raw, client_info, audit=FakeAudit(), ai_client=ai
    )

    meta = store["sess1"]["summary_classifications"]["1"]
    assert meta["classification"]["category"] == "not_mine"
    assert meta["summary_hash"] == summary_hash(structured["1"])
    assert meta["state"] == "CA"
    assert meta["rules_version"] == recompute_rules_version()


class FakeAudit:
    def log_account(self, *a, **k):
        pass

    level = None


def test_custom_letter_uses_cached_classification(monkeypatch):
    summary = {"account_id": "1", "facts_summary": "hi", "claimed_errors": []}
    record = ClassificationRecord(
        summary,
        {
            "category": "goodwill",
            "legal_tag": "TAG",
            "dispute_approach": "A",
            "tone": "T",
        },
        summary_hash(summary),
    )
    monkeypatch.setattr(
        "backend.core.logic.letters.generate_custom_letters.get_neutral_phrase",
        lambda *a, **k: (None, None),
    )
    monkeypatch.setattr(
        "backend.core.logic.letters.generate_custom_letters.generate_letter_with_guardrails",
        lambda *a, **k: ("body", None, None),
    )
    ai = FakeAIClient()
    body = call_gpt_for_custom_letter(
        "C",
        "R",
        "A",
        "1",
        "",
        summary,
        record,
        "CA",
        "sess2",
        None,
        ai,
    )
    assert body == "body"


def test_goodwill_preparation_uses_cached_classification(monkeypatch):
    summary = {"account_id": "1", "facts_summary": "hi", "claimed_errors": []}
    record = ClassificationRecord(
        summary,
        {
            "category": "goodwill",
            "legal_tag": "L",
            "dispute_approach": "D",
            "tone": "T",
        },
        summary_hash(summary),
    )
    monkeypatch.setattr(
        "backend.core.logic.letters.goodwill_preparation.get_neutral_phrase",
        lambda *a, **k: ("NP", "reason"),
    )

    accounts = [{"account_id": "1", "name": "Bank", "account_number": "123"}]
    res = prepare_account_summaries(
        accounts,
        {"1": summary},
        {"1": record},
        "CA",
        "sess3",
        audit=None,
    )
    assert res[0]["dispute_reason"] == "goodwill"
    assert res[0]["neutral_phrase"] == "NP"


def _reload_classifier(monkeypatch, **env):
    for k, v in env.items():
        monkeypatch.setenv(k, str(v))
    import backend.api.config as conf

    importlib.reload(conf)
    import backend.core.logic.strategy.summary_classifier as sc

    importlib.reload(sc)
    sc.reset_cache()
    return sc


def test_cache_disabled(monkeypatch):
    sc = _reload_classifier(monkeypatch, CLASSIFY_CACHE_ENABLED="0")
    ai = FakeAIClient()
    ai.add_response('{"category": "not_mine"}')
    summary = {"account_id": "1", "facts_summary": "a", "claimed_errors": []}
    sc.classify_client_summary(summary, ai_client=ai, session_id="s", account_id="1")
    ai.add_response('{"category": "not_mine"}')
    sc.classify_client_summary(summary, ai_client=ai, session_id="s", account_id="1")
    assert sc.cache_hits() == 0
    assert sc.cache_misses() == 2
    assert len(ai.chat_payloads) == 2


def test_cache_eviction_on_maxsize(monkeypatch):
    sc = _reload_classifier(
        monkeypatch, CLASSIFY_CACHE_MAXSIZE="2", CLASSIFY_CACHE_ENABLED="1"
    )
    summaries = [
        {"account_id": "1", "facts_summary": "a", "claimed_errors": []},
        {"account_id": "2", "facts_summary": "b", "claimed_errors": []},
        {"account_id": "3", "facts_summary": "c", "claimed_errors": []},
    ]
    ai = FakeAIClient()
    for i, summary in enumerate(summaries, 1):
        ai.add_response('{"category": "not_mine"}')
        sc.classify_client_summary(
            summary, ai_client=ai, session_id="s", account_id=str(i)
        )
    assert sc.cache_evictions() == 1
    ai.add_response('{"category": "not_mine"}')
    sc.classify_client_summary(
        summaries[0], ai_client=ai, session_id="s", account_id="1"
    )
    assert len(ai.chat_payloads) == 4


def test_cache_ttl_expiry(monkeypatch):
    sc = _reload_classifier(monkeypatch, CLASSIFY_CACHE_TTL_SEC="1")
    ai = FakeAIClient()
    summary = {"account_id": "1", "facts_summary": "a", "claimed_errors": []}
    ai.add_response('{"category": "not_mine"}')
    sc.classify_client_summary(summary, ai_client=ai, session_id="s", account_id="1")
    time.sleep(1.1)
    ai.add_response('{"category": "not_mine"}')
    sc.classify_client_summary(summary, ai_client=ai, session_id="s", account_id="1")
    assert sc.cache_hits() == 0
    assert sc.cache_misses() == 2
    assert sc.cache_evictions() >= 1
    assert len(ai.chat_payloads) == 2


def test_rules_version_mismatch_triggers_reclassify(monkeypatch):
    sc = _reload_classifier(monkeypatch)
    ai = FakeAIClient()
    summary = {"account_id": "1", "facts_summary": "a", "claimed_errors": []}
    ai.add_response('{"category": "not_mine"}')
    sc.classify_client_summary(
        summary, ai_client=ai, session_id="s", account_id="1", state="CA"
    )
    sc.RULES_VERSION = "different"
    ai.add_response('{"category": "not_mine"}')
    sc.classify_client_summary(
        summary, ai_client=ai, session_id="s", account_id="1", state="CA"
    )
    assert sc.cache_hits() == 0
    assert sc.cache_misses() == 2
    assert len(ai.chat_payloads) == 2


def test_state_variation_uses_separate_cache(monkeypatch):
    sc = _reload_classifier(monkeypatch)
    ai = FakeAIClient()
    summary = {"account_id": "1", "facts_summary": "a", "claimed_errors": []}
    ai.add_response('{"category": "not_mine"}')
    sc.classify_client_summary(
        summary, ai_client=ai, session_id="s", account_id="1", state="CA"
    )
    ai.add_response('{"category": "not_mine"}')
    sc.classify_client_summary(
        summary, ai_client=ai, session_id="s", account_id="1", state="NY"
    )
    sc.classify_client_summary(
        summary, ai_client=ai, session_id="s", account_id="1", state="CA"
    )
    sc.classify_client_summary(
        summary, ai_client=ai, session_id="s", account_id="1", state="NY"
    )
    assert sc.cache_hits() == 2
    assert sc.cache_misses() == 2
    assert len(ai.chat_payloads) == 2
