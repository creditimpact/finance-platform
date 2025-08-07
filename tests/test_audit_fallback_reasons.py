import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from audit import start_audit, clear_audit
from logic.constants import FallbackReason


def test_apply_fallback_tags_logs_keyword_match(tmp_path, monkeypatch):
    import types
    sys.modules['pdfkit'] = types.SimpleNamespace(configuration=lambda **kwargs: None)
    from logic.process_accounts import process_analyzed_report

    audit = start_audit()
    report = {
        "negative_accounts": [
            {"name": "Bad Corp", "bureaus": ["Experian"], "status": "collection"}
        ]
    }
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report))
    process_analyzed_report(path)
    audit_file = audit.save(tmp_path)
    data = json.loads(audit_file.read_text())
    entries = data["accounts"]["Bad Corp"]
    assert any(e.get("fallback_reason") == FallbackReason.KEYWORD_MATCH.value for e in entries)
    clear_audit()


def test_merge_strategy_data_audit_reasons(tmp_path):
    import types
    sys.modules['pdfkit'] = types.SimpleNamespace(configuration=lambda **kwargs: None)
    import main

    audit = start_audit()
    strategy = {
        "accounts": [
            {"name": "Bad Corp", "account_number": "1111", "recommended_action": "foobar"}
        ]
    }
    bureau_data = {
        "Experian": {
            "disputes": [
                {"name": "Bad Corp", "account_number": "1111", "status": "collection"},
                {"name": "No Strat", "account_number": "2222", "status": "chargeoff"},
            ],
            "goodwill": [],
            "high_utilization": [],
        }
    }
    classification_map = {}
    log_list = []
    main.merge_strategy_data(strategy, bureau_data, classification_map, audit, log_list)
    audit_file = audit.save(tmp_path)
    data = json.loads(audit_file.read_text())
    bad_logs = data["accounts"]["Bad Corp"]
    no_logs = data["accounts"]["No Strat"]
    assert any(e.get("fallback_reason") == FallbackReason.UNRECOGNIZED_TAG.value for e in bad_logs)
    assert any(e.get("fallback_reason") == FallbackReason.NO_RECOMMENDATION.value for e in no_logs)
    clear_audit()
