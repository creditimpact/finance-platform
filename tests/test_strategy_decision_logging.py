import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from audit import create_audit_logger
from models.account import Account
from models.strategy import StrategyPlan


def test_strategy_decision_logged_for_all_accounts(tmp_path):
    import types
    sys.modules['pdfkit'] = types.SimpleNamespace(configuration=lambda **kwargs: None)
    from logic.strategy_merger import merge_strategy_data

    audit = create_audit_logger("test")
    strategy = StrategyPlan.from_dict(
        {
            "accounts": [
                {"name": "Bad Corp", "account_number": "1111", "recommended_action": "foobar"},
                {"name": "Good Tag", "account_number": "3333", "recommended_action": "dispute"},
            ]
        }
    )
    bureau_data = {
        "Experian": {
            "disputes": [
                Account.from_dict({"name": "Bad Corp", "account_number": "1111", "status": "collection"}),
                Account.from_dict({"name": "No Strat", "account_number": "2222", "status": "chargeoff"}),
                Account.from_dict({"name": "Good Tag", "account_number": "3333", "status": "open"}),
                Account.from_dict({"name": "No Action", "account_number": "4444", "status": "open"}),
            ],
            "goodwill": [],
            "high_utilization": [],
        }
    }
    classification_map = {}
    merge_strategy_data(strategy, bureau_data, classification_map, audit, [])
    audit_file = audit.save(tmp_path)
    data = json.loads(audit_file.read_text())

    def find_stage(name, stage):
        return next(e for e in data["accounts"][name] if e.get("stage") == stage)

    assert find_stage("Good Tag", "strategy_decision").get("action") == "dispute"
    assert find_stage("Bad Corp", "strategy_decision").get("action") == "dispute"
    assert find_stage("No Strat", "strategy_decision").get("action") == "dispute"
    assert find_stage("No Action", "strategy_decision").get("action") is None

    bad_fallback = find_stage("Bad Corp", "strategy_fallback")
    assert bad_fallback.get("strategist_action") == "foobar"
    assert bad_fallback.get("overrode_strategist") is True

    no_action_fallback = find_stage("No Action", "strategy_fallback")
    assert no_action_fallback.get("strategist_action") is None
    assert no_action_fallback.get("overrode_strategist") is False
