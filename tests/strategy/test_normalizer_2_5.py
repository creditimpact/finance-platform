import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[2]))

from backend.core.logic.strategy.normalizer_2_5 import normalize_and_tag


class DummyRulebook:
    def __init__(self, version: str = "v1") -> None:
        self.version = version


@pytest.fixture
def rulebook() -> DummyRulebook:
    return DummyRulebook("2024-01")


def test_missing_user_statement(rulebook: DummyRulebook) -> None:
    result = normalize_and_tag({}, {}, rulebook)
    assert result["legal_safe_summary"] == "No statement provided"
    assert result["suggested_dispute_frame"] == ""
    assert result["rule_hits"] == []
    assert result["needs_evidence"] == []
    assert result["red_flags"] == []
    assert result["prohibited_admission_detected"] is False
    assert result["rulebook_version"] == "2024-01"


def test_statement_passthrough(rulebook: DummyRulebook) -> None:
    account_cls = {"user_statement_raw": "I paid on time"}
    result = normalize_and_tag(account_cls, {}, rulebook)
    assert result["legal_safe_summary"] == "I paid on time"
    assert result["prohibited_admission_detected"] is False
