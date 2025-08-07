import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from audit import start_audit, clear_audit
from analytics.strategist_failures import tally_fallback_vs_decision


def test_tally_fallback_vs_decision(tmp_path):
    import types
    sys.modules['pdfkit'] = types.SimpleNamespace(configuration=lambda **kwargs: None)
    import main

    audit = start_audit()
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
            ],
            "goodwill": [],
            "high_utilization": [],
        }
    }
    classification_map = {}
    main.merge_strategy_data(strategy, bureau_data, classification_map, audit, [])

    counts = tally_fallback_vs_decision(audit)
    assert counts["strategy_fallback"] == 2
    assert counts["strategy_decision_only"] == 1
    clear_audit()
