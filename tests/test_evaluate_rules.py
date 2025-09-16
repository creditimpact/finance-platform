from backend.core.logic.report_analysis.problem_extractor import (
    evaluate_account_problem,
)


def _reasons(decision: dict) -> list[str]:
    return list(decision.get("problem_reasons") or [])


def test_evaluate_rules_clean_returns_no_reasons():
    fields = {
        "payment_status": "Current",
        "account_status": "Open",
        "past_due_amount": 0.0,
        "balance_owed": 0.0,
        "days_late_7y": 0,
    }
    dec = evaluate_account_problem(fields)
    assert _reasons(dec) == []


def test_evaluate_rules_past_due_flags_and_primary_issue():
    fields = {
        "payment_status": "Current",
        "account_status": "Open",
        "past_due_amount": 120.91,
        "balance_owed": 0.0,
        "days_late_7y": 0,
    }
    dec = evaluate_account_problem(fields)
    reasons = _reasons(dec)
    assert any(r.startswith("past_due_amount:") for r in reasons)
    assert dec.get("primary_issue") == "delinquency"


def test_evaluate_rules_payment_status_tokens():
    fields = {
        "payment_status": "120",
        "account_status": "Open",
        "past_due_amount": 0.0,
        "balance_owed": 0.0,
        "days_late_7y": 0,
    }
    dec = evaluate_account_problem(fields)
    reasons = _reasons(dec)
    assert any(r.startswith("bad_payment_status:") for r in reasons)
    assert dec.get("primary_issue") == "status"


def test_evaluate_rules_charge_off_priority():
    fields = {
        "payment_status": "charge-off",
        "account_status": "Open",
        "past_due_amount": 0.0,
        "balance_owed": 0.0,
        "days_late_7y": 0,
    }
    dec = evaluate_account_problem(fields)
    reasons = _reasons(dec)
    assert any(r.startswith("bad_payment_status:") for r in reasons)
    assert dec.get("primary_issue") == "charge_off"


def test_evaluate_rules_account_status_collections():
    fields = {
        "payment_status": "Current",
        "account_status": "collections",
        "past_due_amount": 0.0,
        "balance_owed": 0.0,
        "days_late_7y": 0,
    }
    dec = evaluate_account_problem(fields)
    reasons = _reasons(dec)
    assert any(r.startswith("bad_account_status:") for r in reasons)
    assert dec.get("primary_issue") == "collection"


def test_evaluate_rules_days_late_history():
    fields = {
        "payment_status": "Current",
        "account_status": "Open",
        "past_due_amount": 0.0,
        "balance_owed": 0.0,
        "days_late_7y": 2,
    }
    dec = evaluate_account_problem(fields)
    reasons = _reasons(dec)
    assert any("late_history: days_late_7y=2" in r for r in reasons)
    assert dec.get("primary_issue") == "late_history"

