import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from backend.audit.audit import create_audit_logger
from backend.core.logic.compliance.constants import (
    FallbackReason,
    StrategistFailureReason,
)
from backend.core.models.account import Account
from backend.core.models.strategy import StrategyPlan


def test_apply_fallback_tags_logs_keyword_match(tmp_path, monkeypatch):
    import types

    sys.modules["pdfkit"] = types.SimpleNamespace(configuration=lambda **kwargs: None)
    from backend.core.logic.report_analysis.process_accounts import process_analyzed_report

    audit = create_audit_logger("test")
    report = {
        "negative_accounts": [
            {"name": "Bad Corp", "bureaus": ["Experian"], "status": "collection"}
        ]
    }
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report))
    process_analyzed_report(path, audit)
    audit_file = audit.save(tmp_path)
    data = json.loads(audit_file.read_text())
    entries = data["accounts"]["Bad Corp"]
    assert any(
        e.get("fallback_reason") == FallbackReason.KEYWORD_MATCH.value for e in entries
    )


def test_merge_strategy_data_audit_reasons(tmp_path):
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
    classification_map = {}
    log_list = []
    merge_strategy_data(strategy, bureau_data, classification_map, audit, log_list)
    audit_file = audit.save(tmp_path)
    data = json.loads(audit_file.read_text())
    bad_logs = data["accounts"]["Bad Corp"]
    no_logs = data["accounts"]["No Strat"]
    empty_logs = data["accounts"]["Empty Action"]

    # Bad Corp: strategist provided unrecognised action
    assert any(
        e.get("fallback_reason") == FallbackReason.UNRECOGNIZED_TAG.value
        for e in bad_logs
    )
    assert any(
        e.get("failure_reason") == StrategistFailureReason.UNRECOGNIZED_FORMAT.value
        for e in bad_logs
        if e.get("stage") == "strategy_fallback"
    )
    fail_entry = next(e for e in bad_logs if e.get("stage") == "strategist_failure")
    fallback_entry = next(e for e in bad_logs if e.get("stage") == "strategy_fallback")
    assert fail_entry.get("raw_action") == "foobar"
    assert fallback_entry.get("raw_action") == "foobar"

    # No Strat: strategist missing entry
    assert any(
        e.get("fallback_reason") == FallbackReason.NO_RECOMMENDATION.value
        for e in no_logs
    )
    assert any(
        e.get("failure_reason") == StrategistFailureReason.MISSING_INPUT.value
        for e in no_logs
        if e.get("stage") == "strategy_fallback"
    )

    # Empty Action: strategist provided entry with empty recommendation
    assert any(
        e.get("fallback_reason") == FallbackReason.NO_RECOMMENDATION.value
        for e in empty_logs
    )
    assert any(
        e.get("failure_reason") == StrategistFailureReason.EMPTY_OUTPUT.value
        for e in empty_logs
        if e.get("stage") == "strategy_fallback"
    )
