import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from backend.audit.audit import create_audit_logger
from backend.core.logic.constants import FallbackReason
from backend.core.models.account import Account
from backend.core.models.strategy import StrategyPlan


def test_strategy_fallback_logs_include_reason_and_override(tmp_path):
    import types

    sys.modules["pdfkit"] = types.SimpleNamespace(configuration=lambda **kwargs: None)
    from backend.core.logic.strategy_merger import merge_strategy_data

    audit = create_audit_logger("test")
    strategy = StrategyPlan.from_dict(
        {
            "accounts": [
                {
                    "name": "Bad Corp",
                    "account_number": "1111",
                    "recommended_action": "foobar",
                }
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
            ],
            "goodwill": [],
            "high_utilization": [],
        }
    }
    classification_map = {}
    merge_strategy_data(strategy, bureau_data, classification_map, audit, [])
    audit_file = audit.save(tmp_path)
    data = json.loads(audit_file.read_text())

    bad_entry = next(
        e for e in data["accounts"]["Bad Corp"] if e.get("stage") == "strategy_fallback"
    )
    no_entry = next(
        e for e in data["accounts"]["No Strat"] if e.get("stage") == "strategy_fallback"
    )

    assert bad_entry.get("fallback_reason") == FallbackReason.UNRECOGNIZED_TAG.value
    assert bad_entry.get("overrode_strategist") is True
    assert bad_entry.get("strategist_action") == "foobar"
    assert no_entry.get("fallback_reason") == FallbackReason.NO_RECOMMENDATION.value
    assert no_entry.get("overrode_strategist") is False
    assert "strategist_action" in no_entry and no_entry.get("strategist_action") is None
