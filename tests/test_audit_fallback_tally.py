import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from backend.audit.audit import create_audit_logger
from backend.analytics.analytics.strategist_failures import tally_fallback_vs_decision
from backend.core.models.account import Account
from backend.core.models.strategy import StrategyPlan


def test_tally_fallback_vs_decision(tmp_path):
    import types

    sys.modules["pdfkit"] = types.SimpleNamespace(configuration=lambda **kwargs: None)
    from backend.core.logic.strategy.strategy_merger import merge_strategy_data

    audit = create_audit_logger("test")
    strategy = StrategyPlan.from_dict(
        {
            "accounts": [
                {
                    "name": "Bad Corp",
                    "account_number": "1111",
                    "recommended_action": "foobar",
                },
                {
                    "name": "Good Tag",
                    "account_number": "3333",
                    "recommended_action": "dispute",
                },
            ]
        }
    )
    bureau_data = {
        "Experian": {
            "disputes": [
                Account.from_dict(
                    {
                        "name": "Bad Corp",
                        "account_number": "1111",
                        "status": "collection",
                    }
                ),
                Account.from_dict(
                    {
                        "name": "No Strat",
                        "account_number": "2222",
                        "status": "chargeoff",
                    }
                ),
                Account.from_dict(
                    {"name": "Good Tag", "account_number": "3333", "status": "open"}
                ),
            ],
            "goodwill": [],
            "high_utilization": [],
        }
    }
    classification_map = {}
    merge_strategy_data(strategy, bureau_data, classification_map, audit, [])

    counts = tally_fallback_vs_decision(audit)
    assert counts["strategy_fallback"] == 2
    assert counts["strategy_decision_only"] == 1
