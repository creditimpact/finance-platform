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
    result = normalize_and_tag(account_cls, {}, rulebook, account_id="acct-1")
    assert (
        result["legal_safe_summary"]
        == "Creditor reports a debt; consumer requests verification."
    )
    assert result["red_flags"] == ["admission_of_debt"]
    assert result["prohibited_admission_detected"] is True
    counters = get_counters()
    assert counters["s2_5_accounts_total"] == 1
    assert counters["s2_5_admissions_detected_total"] == 1
    assert counters["s2_5_rule_hits_total"] == 0
    assert counters["s2_5_needs_evidence_total"] == 0
    assert counters["s2_5_rule_hits_per_account"] == 0
    assert counters["s2_5_latency_ms"] > 0


def test_spanish_admission_detected(rulebook: DummyRulebook) -> None:
    account_cls = {"user_statement_raw": "Pagué tarde en esta cuenta"}
    result = normalize_and_tag(account_cls, {}, rulebook, account_id="acct-2")
    assert (
        result["legal_safe_summary"]
        == "Creditor reports a late payment; consumer requests verification."
    )
    assert result["red_flags"] == ["late_payment"]
    assert result["prohibited_admission_detected"] is True
    counters = get_counters()
    assert counters["s2_5_admissions_detected_total"] == 1


def test_placeholder_handled(rulebook: DummyRulebook) -> None:
    result = normalize_and_tag({}, {}, rulebook, account_id="acct-3")
    assert result["legal_safe_summary"] == "No statement provided"
    assert result["red_flags"] == []
    assert result["prohibited_admission_detected"] is False
    counters = get_counters()
    assert counters.get("s2_5_admissions_detected_total", 0) == 0
    assert counters["s2_5_accounts_total"] == 1


def test_admission_emits_event(rulebook: DummyRulebook, monkeypatch) -> None:
    events = []

    def fake_emit(event, payload):
        events.append((event, payload))

    monkeypatch.setattr(
        "backend.core.logic.strategy.normalizer_2_5.emit_event", fake_emit
    )
    account_cls = {"user_statement_raw": "I owe on account 123456789012"}
    normalize_and_tag(account_cls, {}, rulebook, account_id="123456789012")
    assert events
    event, payload = events[0]
    assert event == "admission_neutralized"
    assert payload["account_id"] == "[REDACTED]"
    assert payload["raw_statement"] == "I owe on account [REDACTED]"
    assert (
        payload["summary"]
        == "Creditor reports a debt; consumer requests verification."
    )
