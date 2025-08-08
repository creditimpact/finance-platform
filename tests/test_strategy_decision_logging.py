import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from audit import create_audit_logger


def test_strategy_decision_logged_for_all_accounts(tmp_path):
    import types
    sys.modules['pdfkit'] = types.SimpleNamespace(configuration=lambda **kwargs: None)
    import main

    audit = create_audit_logger("test")
    strategy = {
        "accounts": [
            {"name": "Bad Corp", "account_number": "1111", "recommended_action": "foobar"},
            {"name": "Good Tag", "account_number": "3333", "recommended_action": "dispute"},
        ]
    }
    bureau_data = {
        "Experian": {
            "disputes": [
                {"name": "Bad Corp", "account_number": "1111", "status": "collection"},
                {"name": "No Strat", "account_number": "2222", "status": "chargeoff"},
                {"name": "Good Tag", "account_number": "3333", "status": "open"},
                {"name": "No Action", "account_number": "4444", "status": "open"},
            ],
            "goodwill": [],
            "high_utilization": [],
        }
    }
    classification_map = {}
    main.merge_strategy_data(strategy, bureau_data, classification_map, audit, [])
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
