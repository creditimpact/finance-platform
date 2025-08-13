import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from backend.core.models.strategy import StrategyPlan


def test_parses_legacy_strategy():
    data = {
        "accounts": [
            {
                "account_id": "1",
                "name": "Account A",
                "account_number": "1234",
                "recommendation": {"recommended_action": "Dispute"},
            }
        ]
    }
    plan = StrategyPlan.from_dict(data)
    item = plan.accounts[0]
    assert item.account_id == "1"
    assert item.legal_safe_summary is None
    assert item.rule_hits == []


def test_parses_new_fields():
    data = {
        "accounts": [
            {
                "account_id": "1",
                "name": "Account A",
                "account_number": "1234",
                "legal_safe_summary": "summary",
                "suggested_dispute_frame": "frame",
                "rule_hits": ["R1"],
                "needs_evidence": ["E1"],
                "red_flags": ["F1"],
                "recommendation": {"recommended_action": "Dispute"},
            }
        ]
    }
    plan = StrategyPlan.from_dict(data)
    item = plan.accounts[0]
    assert item.legal_safe_summary == "summary"
    assert item.suggested_dispute_frame == "frame"
    assert item.rule_hits == ["R1"]
    assert item.needs_evidence == ["E1"]
    assert item.red_flags == ["F1"]
