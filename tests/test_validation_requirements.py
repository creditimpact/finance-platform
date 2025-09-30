import pytest

from backend.core.logic.consistency import compute_field_consistency
from backend.core.logic.validation_requirements import build_validation_requirements


@pytest.fixture
def account_with_histories():
    return {
        "transunion": {
            "two_year_payment_history": ["CO", "30", "OK", "120"],
            "seven_year_history": {"late30": "2", "co_count": 1},
        },
        "experian": {
            "two_year_payment_history": ["OK", "OK", "OK"],
            "seven_year_history": {"late30": 0, "co_count": 0},
        },
        "equifax": {
            "two_year_payment_history": [],
            "seven_year_history": {"late30": 0, "co_count": 0},
        },
    }


@pytest.fixture
def account_number_masking_only():
    return {
        "transunion": {"account_number_display": "349991234567"},
        "experian": {"account_number_display": "-34999***********4567"},
        "equifax": {},
    }


@pytest.fixture
def account_money_mismatch():
    return {
        "transunion": {"high_balance": 19944},
        "experian": {"high_balance": "$0"},
        "equifax": {},
    }


@pytest.fixture
def account_date_mismatch():
    return {
        "transunion": {"date_of_last_activity": "16.8.2022"},
        "experian": {"date_of_last_activity": "1.8.2023"},
        "equifax": {},
    }


@pytest.fixture
def account_soft_semantics():
    return {
        "transunion": {"creditor_type": "Bank Credit Cards"},
        "experian": {"creditor_type": "All Banks"},
        "equifax": {"creditor_type": "bank credit cards"},
    }


def _requirements_by_field(bureaus):
    requirements, inconsistencies, _ = build_validation_requirements(bureaus)
    by_field = {entry["field"]: entry for entry in requirements}
    return by_field, inconsistencies


def test_account_number_last4_mismatch_is_strong():
    bureaus = {
        "transunion": {"account_number_display": "****9992"},
        "experian": {"account_number_display": "4999"},
        "equifax": {"account_number_display": "acct 4999"},
    }

    requirements, inconsistencies = _requirements_by_field(bureaus)

    rule = requirements["account_number_display"]
    assert rule["strength"] == "strong"
    assert rule["ai_needed"] is False

    normalized = inconsistencies["account_number_display"]["normalized"]
    assert normalized["transunion"]["last4"] == "9992"
    assert normalized["experian"]["last4"] == "4999"
def test_histories_require_strong_policy(account_with_histories):
    requirements, inconsistencies = _requirements_by_field(account_with_histories)

    assert "two_year_payment_history" in requirements
    history_rule = requirements["two_year_payment_history"]
    assert history_rule["strength"] == "strong"
    assert history_rule["ai_needed"] is False
    assert history_rule["min_days"] == 18
    assert history_rule["documents"] == [
        "monthly_statements_2y",
        "internal_payment_history",
    ]

    history_norm = inconsistencies["two_year_payment_history"]["normalized"]
    assert history_norm["transunion"]["tokens"] == ["CO", "30", "OK", "120"]
    assert history_norm["transunion"]["counts"]["CO"] == 1
    assert history_norm["transunion"]["counts"]["late30"] == 1
    assert history_norm["experian"]["tokens"] == ["OK", "OK", "OK"]
    assert history_norm["equifax"]["tokens"] == []

    assert "seven_year_history" in requirements
    seven_rule = requirements["seven_year_history"]
    assert seven_rule["strength"] == "strong"
    assert seven_rule["ai_needed"] is False
    assert seven_rule["min_days"] == 25
    assert seven_rule["documents"] == [
        "cra_report_7y",
        "cra_audit_logs",
        "collection_history",
    ]

    seven_norm = inconsistencies["seven_year_history"]["normalized"]
    assert seven_norm["transunion"]["late30"] == 2
    assert seven_norm["transunion"]["late90"] == 1
    assert seven_norm["experian"]["late30"] == 0
    assert seven_norm["equifax"]["late30"] == 0


def test_money_mismatch_is_strong(account_money_mismatch):
    requirements, _ = _requirements_by_field(account_money_mismatch)

    assert "high_balance" in requirements
    rule = requirements["high_balance"]
    assert rule["strength"] == "strong"
    assert rule["ai_needed"] is False


def test_dates_are_normalized(account_date_mismatch):
    requirements, inconsistencies = _requirements_by_field(account_date_mismatch)

    assert "date_of_last_activity" in requirements
    rule = requirements["date_of_last_activity"]
    assert rule["strength"] == "strong"
    assert rule["ai_needed"] is False

    normalized = inconsistencies["date_of_last_activity"]["normalized"]
    assert normalized["transunion"] == "2022-08-16"
    assert normalized["experian"] == "2023-08-01"


def test_account_number_masking_needs_soft_review(account_number_masking_only):
    requirements, inconsistencies = _requirements_by_field(account_number_masking_only)

    assert "account_number_display" in requirements
    rule = requirements["account_number_display"]
    assert rule["strength"] == "soft"
    assert rule["ai_needed"] is True
    assert rule["bureaus"] == ["equifax", "experian", "transunion"]

    field_consistency = compute_field_consistency(account_number_masking_only)
    assert field_consistency["account_number_display"]["consensus"] == "majority"
    normalized = inconsistencies["account_number_display"]["normalized"]
    assert normalized["transunion"]["last4"] == "4567"
    assert normalized["experian"]["last4"] == "4567"
    assert normalized["equifax"]["last4"] is None


def test_soft_semantics_require_ai(account_soft_semantics):
    requirements, _ = _requirements_by_field(account_soft_semantics)

    assert "creditor_type" in requirements
    rule = requirements["creditor_type"]
    assert rule["strength"] == "soft"
    assert rule["ai_needed"] is True


def test_sanity_example_strong_and_soft_requirements():
    bureaus = {
        "transunion": {
            "two_year_payment_history": [],
            "seven_year_history": {"late30": 0, "late60": 0, "late90": 0},
            "date_of_last_activity": "16.8.2022",
            "dispute_status": "Not disputed",
            "account_number_display": "****1234",
            "creditor_remarks": "Account transferred to recovery",
        },
        "experian": {
            "two_year_payment_history": ["CO"] * 24,
            "seven_year_history": {"late30": 0, "late60": 0, "late90": 0},
            "date_of_last_activity": "1.8.2023",
            "dispute_status": "not disputed",
            "account_number_display": "1234",
            "creditor_remarks": "Account closed at customer request",
        },
        "equifax": {
            "two_year_payment_history": "",
            "seven_year_history": {"late30": 0, "late60": 0, "late90": 28},
            "date_of_last_activity": "1.11.2022",
            "dispute_status": "Account disputed",
            "account_number_display": "xxxx-1234",
            "creditor_remarks": "Consumer states balance incorrect",
        },
    }

    requirements, inconsistencies = _requirements_by_field(bureaus)

    assert "two_year_payment_history" in requirements
    two_year_rule = requirements["two_year_payment_history"]
    assert two_year_rule["strength"] == "strong"
    assert two_year_rule["ai_needed"] is False
    assert two_year_rule["min_days"] == 18
    assert two_year_rule["documents"] == [
        "monthly_statements_2y",
        "internal_payment_history",
    ]
    two_year_consensus = inconsistencies["two_year_payment_history"]["consensus"]
    assert two_year_consensus in {"majority", "split"}
    two_year_norm = inconsistencies["two_year_payment_history"]["normalized"]
    assert two_year_norm["experian"]["counts"]["CO"] == 24
    assert two_year_norm["transunion"]["tokens"] == []

    assert "seven_year_history" in requirements
    seven_rule = requirements["seven_year_history"]
    assert seven_rule["strength"] == "strong"
    assert seven_rule["ai_needed"] is False
    seven_norm = inconsistencies["seven_year_history"]["normalized"]
    assert seven_norm["equifax"]["late90"] == 28

    assert "date_of_last_activity" in requirements
    assert requirements["date_of_last_activity"]["strength"] == "strong"

    assert "dispute_status" in requirements
    assert requirements["dispute_status"]["strength"] == "strong"

    assert "creditor_remarks" in requirements
    creditor_rule = requirements["creditor_remarks"]
    assert creditor_rule["strength"] == "soft"
    assert creditor_rule["ai_needed"] is True

    assert "account_number_display" not in requirements
