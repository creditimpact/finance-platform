import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[2]))

from backend.core.logic.strategy.normalizer_2_5 import normalize_and_tag
from backend.analytics.analytics_tracker import get_counters, reset_counters


class DummyRulebook:
    def __init__(self, version: str = "2024-01") -> None:
        self.version = version


@pytest.fixture(autouse=True)
def clear_metrics():
    reset_counters()
    yield
    reset_counters()


@pytest.fixture
def rulebook() -> DummyRulebook:
    return DummyRulebook()


def test_admission_neutralized_with_red_flags(rulebook: DummyRulebook) -> None:
    account_cls = {"user_statement_raw": "I owe them money"}
    result = normalize_and_tag(account_cls, {}, rulebook)
    assert (
        result["legal_safe_summary"]
        == "Creditor reports a debt; consumer requests verification."
    )
    assert result["red_flags"] == ["admission_of_debt"]
    assert result["prohibited_admission_detected"] is True
    counters = get_counters()
    assert counters["stage_2_5.admission_neutralized_total"] == 1
    assert counters["stage_2_5.rules_applied"] == 1


def test_spanish_admission_detected(rulebook: DummyRulebook) -> None:
    account_cls = {"user_statement_raw": "Pagué tarde en esta cuenta"}
    result = normalize_and_tag(account_cls, {}, rulebook)
    assert (
        result["legal_safe_summary"]
        == "Creditor reports a late payment; consumer requests verification."
    )
    assert result["red_flags"] == ["late_payment"]
    assert result["prohibited_admission_detected"] is True
    counters = get_counters()
    assert counters["stage_2_5.admission_neutralized_total"] == 1


def test_placeholder_handled(rulebook: DummyRulebook) -> None:
    result = normalize_and_tag({}, {}, rulebook)
    assert result["legal_safe_summary"] == "No statement provided"
    assert result["red_flags"] == []
    assert result["prohibited_admission_detected"] is False
    counters = get_counters()
    assert counters.get("stage_2_5.admission_neutralized_total", 0) == 0
    assert counters["stage_2_5.rules_applied"] == 1


def test_admission_emits_event(rulebook: DummyRulebook, monkeypatch) -> None:
    events = []

    def fake_emit(event, payload):
        events.append((event, payload))

    monkeypatch.setattr(
        "backend.core.logic.strategy.normalizer_2_5.emit_event", fake_emit
    )
    account_cls = {"user_statement_raw": "I owe them money"}
    normalize_and_tag(account_cls, {}, rulebook)
    assert events
    event, payload = events[0]
    assert event == "admission_neutralized"
    assert payload["raw_statement"] == "I owe them money"
    assert (
        payload["summary"]
        == "Creditor reports a debt; consumer requests verification."
    )
