"""Tests for field consistency logic."""

from backend.core.logic.consistency import compute_field_consistency


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
