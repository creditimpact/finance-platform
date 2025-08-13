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
    account_cls = {"user_statement_raw": "I was late paying this account"}
    result = normalize_and_tag(account_cls, {}, rulebook)
    assert result["legal_safe_summary"] == "The consumer was late paying this account"
    assert result["red_flags"] == ["late_payment"]
    counters = get_counters()
    assert counters["stage_2_5.admission_neutralized"] == 1
    assert counters["stage_2_5.rules_applied"] == 1


def test_placeholder_handled(rulebook: DummyRulebook) -> None:
    result = normalize_and_tag({}, {}, rulebook)
    assert result["legal_safe_summary"] == "No statement provided"
    assert result["red_flags"] == []
    counters = get_counters()
    assert counters.get("stage_2_5.admission_neutralized", 0) == 0
    assert counters["stage_2_5.rules_applied"] == 1
