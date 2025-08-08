import sys
import types

sys.modules['pdfkit'] = types.SimpleNamespace(configuration=lambda **kwargs: None)

import main
from audit import create_audit_logger
from analytics.strategist_failures import tally_failure_reasons
from logic.constants import StrategistFailureReason
from models.account import Account
from models.strategy import StrategyPlan


def test_tally_failure_reasons():
    audit = create_audit_logger("test")

    strategy = StrategyPlan.from_dict(
        {
            "accounts": [
                {"name": "Bad Corp", "account_number": "1111", "recommended_action": "foobar"},
                {"name": "Empty Action", "account_number": "3333"},
            ]
        }
    )
    bureau_data = {
        "Experian": {
            "disputes": [
                Account.from_dict({"name": "Bad Corp", "account_number": "1111", "status": "collection"}),
                Account.from_dict({"name": "No Strat", "account_number": "2222", "status": "chargeoff"}),
                Account.from_dict({"name": "Empty Action", "account_number": "3333", "status": "repossession"}),
            ],
            "goodwill": [],
            "high_utilization": [],
        }
    }

    main.merge_strategy_data(strategy, bureau_data, {}, audit, [])

    counts = tally_failure_reasons(audit)

    accounts = audit.data["accounts"]
    assert any(e["stage"] == "strategy_decision" for e in accounts["No Strat"])
    assert any(e["stage"] == "strategy_decision" for e in accounts["Empty Action"])

    expected = {
        StrategistFailureReason.UNRECOGNIZED_FORMAT.value: 1,
        StrategistFailureReason.MISSING_INPUT.value: 1,
        StrategistFailureReason.EMPTY_OUTPUT.value: 1,
    }

    assert counts == expected
