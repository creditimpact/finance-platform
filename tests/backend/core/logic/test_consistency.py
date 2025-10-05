"""Tests for field consistency logic."""

from backend.core.logic.consistency import compute_field_consistency
from backend.core.logic.reason_classifier import classify_reason


def test_missing_bureau_is_recorded_and_breaks_unanimous_consensus() -> None:
    bureaus_json = {
        "transunion": {"account_status": "open"},
        "experian": {"account_status": ""},
        "equifax": {"account_status": "open"},
    }

    consistency = compute_field_consistency(bureaus_json)
    account_status = consistency["account_status"]

    assert account_status["missing_bureaus"] == ["experian"]
    assert account_status["consensus"] == "majority"
    assert account_status["disagreeing_bureaus"] == ["experian"]


def test_history_values_capture_raw_and_missing_bureau_listed() -> None:
    bureaus_json = {
        "transunion": {},
        "experian": {},
        "equifax": {},
        "two_year_payment_history": {
            "transunion": ["30"],
            "experian": ["OK"],
            "equifax": None,
        },
    }

    consistency = compute_field_consistency(bureaus_json)
    history = consistency["two_year_payment_history"]

    assert history["raw"] == {
        "transunion": ["30"],
        "experian": ["OK"],
        "equifax": None,
    }
    assert history["missing_bureaus"] == ["equifax"]
    assert history["consensus"] == "split"


def test_empty_history_entries_are_treated_as_missing() -> None:
    bureaus_json = {
        "transunion": {"two_year_payment_history": {}},
        "experian": {"two_year_payment_history": ["30", "CO"]},
        "equifax": {"two_year_payment_history": []},
    }

    consistency = compute_field_consistency(bureaus_json)
    history = consistency["two_year_payment_history"]

    assert history["normalized"]["transunion"] is None
    assert history["normalized"]["equifax"] is None
    assert history["missing_bureaus"] == ["equifax", "transunion"]
    assert history["consensus"] == "majority"

    reason = classify_reason(history["normalized"])
    assert reason["reason_code"] == "C2_ONE_MISSING"
    assert reason["is_mismatch"] is False


def test_empty_seven_year_history_entries_are_missing() -> None:
    bureaus_json = {
        "transunion": {"seven_year_history": {}},
        "experian": {"seven_year_history": {"30": 1}},
        "equifax": {"seven_year_history": {"late90": 0}},
    }

    consistency = compute_field_consistency(bureaus_json)
    history = consistency["seven_year_history"]

    assert history["normalized"]["transunion"] is None
    assert history["normalized"]["equifax"] is None
    assert history["missing_bureaus"] == ["equifax", "transunion"]

    reason = classify_reason(history["normalized"])
    assert reason["reason_code"] == "C2_ONE_MISSING"
    assert reason["is_mismatch"] is False
