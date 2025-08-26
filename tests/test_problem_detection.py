from backend.core.logic.report_analysis.problem_detection import evaluate_account_problem


def test_flags_late_payments_and_past_due():
    acct1 = {"late_payments": {"Equifax": {"30": 1}}}
    v1 = evaluate_account_problem(acct1)
    assert v1["is_problem"] is True
    assert v1["primary_issue"] == "unknown"
    assert v1["problem_reasons"] == ["late_payment: 1×30 on Equifax"]

    acct2 = {"past_due_amount": 50}
    v2 = evaluate_account_problem(acct2)
    assert v2["is_problem"] is True
    assert v2["primary_issue"] == "unknown"
    assert "past_due_amount" in v2["problem_reasons"]


def test_clean_account_not_flagged():
    clean = {"account_status": "Open", "payment_status": "Pays as agreed"}
    v = evaluate_account_problem(clean)
    assert v["is_problem"] is False
    assert v["primary_issue"] == "unknown"
    assert v["problem_reasons"] == []
