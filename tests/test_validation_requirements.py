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
    requirements, inconsistencies = build_validation_requirements(bureaus)
    by_field = {entry["field"]: entry for entry in requirements}
    return by_field, inconsistencies


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
    assert history_norm["transunion"]["codes"] == ("CO", "30", "OK", "120")
    assert history_norm["transunion"]["summary"]["co_count"] == 1
    assert history_norm["transunion"]["summary"]["late30"] == 1
    assert history_norm["experian"]["codes"] == ("OK", "OK", "OK")
    assert history_norm["equifax"] is None

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
    assert seven_norm["transunion"]["co_count"] == 1
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


def test_account_number_masking_is_unanimous(account_number_masking_only):
    requirements, _ = _requirements_by_field(account_number_masking_only)

    assert requirements == {}

    field_consistency = compute_field_consistency(account_number_masking_only)
    assert (
        field_consistency["account_number_display"]["consensus"] == "unanimous"
    )


def test_soft_semantics_require_ai(account_soft_semantics):
    requirements, _ = _requirements_by_field(account_soft_semantics)

    assert "creditor_type" in requirements
    rule = requirements["creditor_type"]
    assert rule["strength"] == "soft"
    assert rule["ai_needed"] is True
