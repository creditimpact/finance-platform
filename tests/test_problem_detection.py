from backend.core.logic.report_analysis.problem_detection import evaluate_account_problem


def test_tier1_collection_from_status():
    acct = {"account_status": "COLLECTION", "credit_limit": 1000, "balance_owed": 100}
    v = evaluate_account_problem(acct)
    assert v["is_problem"] is True
    assert v["confidence_hint"]["tier"] == 1
    assert v["primary_issue"] == "collection"
    assert any(r.startswith("account_status:") for r in v["problem_reasons"])


def test_tier2_serious_delinquency_from_payment_status():
    acct = {"payment_status": "120 days past due"}
    v = evaluate_account_problem(acct)
    assert v["is_problem"] is True
    assert v["primary_issue"] == "serious_delinquency"
    assert v["confidence_hint"]["tier"] == 2


def test_tier3_from_account_rating_only():
    acct = {"account_rating": "Derogatory"}
    v = evaluate_account_problem(acct)
    assert v["is_problem"] is True
    assert v["primary_issue"] == "potential_derogatory"
    assert v["confidence_hint"]["tier"] == 3


def test_utilization_is_supporting_not_problem():
    acct = {"balance_owed": 950, "credit_limit": 1000}
    v = evaluate_account_problem(acct)
    assert v["is_problem"] is False
    assert "utilization" in v["supporting"]
    assert any("utilization" in r for r in v["problem_reasons"])
