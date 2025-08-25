import backend.core.logic.report_analysis.report_postprocessing as rp
from backend.core.logic.report_analysis import analyze_report as ar


def test_parser_co_grid_goes_negative():
    result = {"all_accounts": []}
    history = {"bad bank": {"Experian": {"30": 1}}}
    raw_map = {"bad bank": "Bad Bank"}
    grid_map = {"bad bank": {"Experian": "OK CO"}}
    rp._inject_missing_late_accounts(result, history, raw_map, grid_map)
    negatives, open_issues = ar._split_account_buckets(result["all_accounts"])
    assert [a["name"] for a in negatives] == ["Bad Bank"]
    assert not open_issues


def test_parser_open_lates_go_open_issues():
    result = {"all_accounts": []}
    history = {"good bank": {"Experian": {"30": 1}}}
    raw_map = {"good bank": "Good Bank"}
    rp._inject_missing_late_accounts(result, history, raw_map, {})
    acc = result["all_accounts"][0]
    acc["account_status"] = "Open"
    acc["payment_status"] = "Open"
    acc["payment_statuses"] = {"Experian": "Open"}
    negatives, open_issues = ar._split_account_buckets(result["all_accounts"])
    assert not negatives
    assert [a["name"] for a in open_issues] == ["Good Bank"]
