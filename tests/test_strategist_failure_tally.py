import sys
import types

sys.modules['pdfkit'] = types.SimpleNamespace(configuration=lambda **kwargs: None)

import main
from audit import start_audit, clear_audit
from analytics.strategist_failures import tally_failure_reasons
from logic.constants import StrategistFailureReason


def test_tally_failure_reasons():
    audit = start_audit()

    strategy = {
        "accounts": [
            {"name": "Bad Corp", "account_number": "1111", "recommended_action": "foobar"},
            {"name": "Empty Action", "account_number": "3333"},
        ]
    }
    bureau_data = {
        "Experian": {
            "disputes": [
                {"name": "Bad Corp", "account_number": "1111", "status": "collection"},
                {"name": "No Strat", "account_number": "2222", "status": "chargeoff"},
                {"name": "Empty Action", "account_number": "3333", "status": "repossession"},
            ],
            "goodwill": [],
            "high_utilization": [],
        }
    }

    main.merge_strategy_data(strategy, bureau_data, {}, audit, [])

    counts = tally_failure_reasons(audit)

    expected = {
        StrategistFailureReason.UNRECOGNIZED_FORMAT.value: 1,
        StrategistFailureReason.MISSING_INPUT.value: 1,
        StrategistFailureReason.EMPTY_OUTPUT.value: 1,
    }

    assert counts == expected
    clear_audit()
