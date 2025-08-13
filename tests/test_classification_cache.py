import time

import time

from backend.core.orchestrators import classify_client_responses
from backend.core.logic.letters.generate_custom_letters import (
    call_gpt_for_custom_letter,
)
from backend.core.logic.letters.goodwill_preparation import prepare_account_summaries
from backend.core.logic.strategy.summary_classifier import summary_hash
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

    monkeypatch.setattr(
        "backend.api.session_manager.update_session", fake_update
    )
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


class FakeAudit:
    def log_account(self, *a, **k):
        pass

    level = None


def test_custom_letter_uses_cached_classification(monkeypatch):
    session_id = "sess2"
    summary = {"account_id": "1", "facts_summary": "hi", "claimed_errors": []}
    h = summary_hash(summary)
    store = {
        session_id: {
            "summary_classifications": {
                "1": {
                    "summary_hash": h,
                    "classified_at": time.time(),
                    "classification": {"category": "goodwill", "legal_tag": "TAG", "dispute_approach": "A", "tone": "T"},
                }
            }
        }
    }

    monkeypatch.setattr(
        "backend.api.session_manager.get_session", lambda sid: store.get(sid)
    )
    monkeypatch.setattr(
        "backend.api.session_manager.update_session", lambda *a, **k: None
    )
    monkeypatch.setattr(
        "backend.core.logic.letters.generate_custom_letters.classify_client_summary",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not call")),
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
        "C", "R", "A", "1", "", summary, "CA", session_id, None, ai
    )
    assert body == "body"


def test_goodwill_preparation_uses_cached_classification(monkeypatch):
    session_id = "sess3"
    summary = {"account_id": "1", "facts_summary": "hi", "claimed_errors": []}
    h = summary_hash(summary)
    store = {
        session_id: {
            "summary_classifications": {
                "1": {
                    "summary_hash": h,
                    "classified_at": time.time(),
                    "classification": {
                        "category": "goodwill",
                        "legal_tag": "L",
                        "dispute_approach": "D",
                        "tone": "T",
                    },
                }
            }
        }
    }

    monkeypatch.setattr(
        "backend.core.logic.letters.goodwill_preparation.get_session",
        lambda sid: store.get(sid),
    )
    monkeypatch.setattr(
        "backend.core.logic.letters.goodwill_preparation.update_session",
        lambda *a, **k: None,
    )
    monkeypatch.setattr(
        "backend.core.logic.letters.goodwill_preparation.classify_client_summary",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not call")),
    )
    monkeypatch.setattr(
        "backend.core.logic.letters.goodwill_preparation.get_neutral_phrase",
        lambda *a, **k: ("NP", "reason"),
    )

    accounts = [{"account_id": "1", "name": "Bank", "account_number": "123"}]
    res = prepare_account_summaries(
        accounts, {"1": summary}, "CA", session_id, audit=None, ai_client=FakeAIClient()
    )
    assert res[0]["dispute_reason"] == "goodwill"
    assert res[0]["neutral_phrase"] == "NP"
