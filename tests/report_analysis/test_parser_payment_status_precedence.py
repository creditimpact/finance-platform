from backend.core.logic.report_analysis import analyze_report as ar


def test_parser_payment_status_precedence():
    result = {}
    history = {"bad": {"Experian": {"30": 1}}}
    raw_map = {"bad": "Bad Bank"}

    ar._inject_missing_late_accounts(result, history, raw_map, {})

    payment_statuses = {"bad": {"Experian": "collection/chargeoff"}}
    remarks = {"bad": ""}
    payment_status_raw = {}

    ar._attach_parser_signals(
        result["all_accounts"],
        payment_statuses,
        remarks,
        payment_status_raw,
    )

    acc = result["all_accounts"][0]
    ar._assign_issue_types(acc)

    assert acc["payment_statuses"]
    assert acc["primary_issue"] in ("collection", "charge_off")
    assert acc["issue_types"][0] == acc["primary_issue"]
    assert "late_payment" in acc["issue_types"]
