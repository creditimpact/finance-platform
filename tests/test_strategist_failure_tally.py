import sys
import types

sys.modules["pdfkit"] = types.SimpleNamespace(configuration=lambda **kwargs: None)

from audit import create_audit_logger  # noqa: E402
from analytics.strategist_failures import tally_failure_reasons  # noqa: E402
from logic.constants import StrategistFailureReason  # noqa: E402
from logic.strategy_merger import merge_strategy_data  # noqa: E402
from models.account import Account  # noqa: E402
from models.strategy import StrategyPlan  # noqa: E402


def test_tally_failure_reasons():
    audit = create_audit_logger("test")

    strategy = StrategyPlan.from_dict(
        {
            "accounts": [
                {
                    "name": "Bad Corp",
                    "account_number": "1111",
                    "recommended_action": "foobar",
                },
                {"name": "Empty Action", "account_number": "3333"},
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
                    {
                        "name": "Empty Action",
                        "account_number": "3333",
                        "status": "repossession",
                    }
                ),
            ],
            "goodwill": [],
            "high_utilization": [],
        }
    }

    merge_strategy_data(strategy, bureau_data, {}, audit, [])

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
