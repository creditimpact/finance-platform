import pytest

from backend.core.logic.letters.generate_custom_letters import call_gpt_for_custom_letter
from backend.core.logic.guardrails import generate_letter_with_guardrails
from backend.analytics.analytics_tracker import get_counters, reset_counters
from tests.helpers.fake_ai_client import FakeAIClient


def _patch_flag(monkeypatch, value: bool) -> None:
    monkeypatch.setattr(
        "backend.api.config.ALLOW_CUSTOM_LETTERS_WITHOUT_STRATEGY", value
    )
    monkeypatch.setattr(
        "backend.core.logic.letters.generate_custom_letters.api_config.ALLOW_CUSTOM_LETTERS_WITHOUT_STRATEGY",
        value,
    )
    monkeypatch.setattr(
        "backend.core.logic.guardrails.api_config.ALLOW_CUSTOM_LETTERS_WITHOUT_STRATEGY",
        value,
    )


def test_call_gpt_requires_strategy(monkeypatch):
    _patch_flag(monkeypatch, False)
    fake = FakeAIClient()
    res = call_gpt_for_custom_letter(
        "Client",
        "Recipient",
        "Account",
        "1",
        "",
        {},
        None,
        "CA",
        "sess",
        audit=None,
        ai_client=fake,
    )
    assert res == "strategy_context_required"


def test_guardrails_requires_strategy(monkeypatch):
    _patch_flag(monkeypatch, False)
    fake = FakeAIClient()
    res, viol, iters = generate_letter_with_guardrails(
        "prompt", "CA", {}, "sess", "custom", ai_client=fake
    )
    assert res == "strategy_context_required"
    assert viol == []
    assert iters == 0


def test_flag_allows_call_gpt_without_strategy(monkeypatch):
    events = []
    monkeypatch.setattr(
        "backend.core.logic.letters.generate_custom_letters.emit_event",
        lambda e, p: events.append((e, p)),
    )
    monkeypatch.setattr(
        "backend.core.logic.guardrails.emit_event",
        lambda e, p: events.append((e, p)),
    )
    _patch_flag(monkeypatch, True)
    reset_counters()
    fake = FakeAIClient()
    fake.add_chat_response("letter body")
    res = call_gpt_for_custom_letter(
        "Client",
        "Recipient",
        "Account",
        "1",
        "",
        {},
        None,
        "CA",
        "sess",
        audit=None,
        ai_client=fake,
    )
    assert res.startswith("letter body")
    assert any(e == "strategy_applied" and not p["strategy_applied"] for e, p in events)
    counters = get_counters()
    assert counters["letters_without_strategy_context"] == 1


def test_flag_allows_guardrails_without_strategy(monkeypatch):
    events = []
    monkeypatch.setattr(
        "backend.core.logic.guardrails.emit_event", lambda e, p: events.append((e, p))
    )
    _patch_flag(monkeypatch, True)
    reset_counters()
    fake = FakeAIClient()
    fake.add_chat_response("body")
    res, viol, _ = generate_letter_with_guardrails(
        "prompt", "CA", {}, "sess", "custom", ai_client=fake
    )
    assert res.startswith("body")
    assert any(e == "strategy_applied" and not p["strategy_applied"] for e, p in events)
    counters = get_counters()
    assert counters["letters_without_strategy_context"] == 1
