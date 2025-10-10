"""Tests for field consistency logic."""

import backend.core.logic.consistency as consistency_mod
import pytest

from backend.core.logic.consistency import (
    compute_field_consistency,
    compute_inconsistent_fields,
)
from backend.core.logic.context import override_validation_context
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


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Paid-as-Agreed", "current"),
        ("Paid on time", "current"),
        ("CHG OFF", "charge off"),
        ("charged off", "charge off"),
        ("chargeoff", "charge off"),
        ("in collections", "collections"),
        ("collection", "collections"),
        ("30-day late", "30 days late"),
        ("30d late", "30 days late"),
        ("late 60", "60 days late"),
        ("90d late", "90 days late"),
    ],
)
def test_account_rating_aliases(raw: str, expected: str) -> None:
    bureaus_json = {"transunion": {"account_rating": raw}}

    consistency = compute_field_consistency(bureaus_json)
    normalized = consistency["account_rating"]["normalized"]["transunion"]

    assert normalized == expected


def test_account_rating_synonyms_produce_unanimous_consensus() -> None:
    bureaus_json = {
        "transunion": {"account_rating": "Paid as agreed"},
        "experian": {"account_rating": "CURRENT"},
        "equifax": {"account_rating": "paid on time"},
    }

    consistency = compute_field_consistency(bureaus_json)
    normalized = consistency["account_rating"]["normalized"]

    assert normalized["transunion"] == "current"
    assert normalized["experian"] == "current"
    assert normalized["equifax"] == "current"
    assert consistency["account_rating"]["consensus"] == "unanimous"


def test_account_rating_alias_table_matches_expected_values() -> None:
    assert consistency_mod._ACCOUNT_RATING_ALIASES == {
        "paid as agreed": "current",
        "pays as agreed": "current",
        "in good standing": "current",
        "up to date": "current",
        "paid on time": "current",
        "charged off": "charge off",
        "chg off": "charge off",
        "chargeoff": "charge off",
        "in collections": "collections",
        "sent to collections": "collections",
        "collection": "collections",
        "30 day late": "30 days late",
        "30d late": "30 days late",
        "late 30": "30 days late",
        "60 day late": "60 days late",
        "60d late": "60 days late",
        "late 60": "60 days late",
        "90 day late": "90 days late",
        "90d late": "90 days late",
        "late 90": "90 days late",
    }


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


def test_account_number_display_missing_masks_are_treated_as_missing() -> None:
    bureaus_json = {
        "transunion": {"account_number_display": "****"},
        "experian": {"account_number_display": None},
        "equifax": {"account_number_display": "****1234"},
    }

    consistency = compute_field_consistency(bureaus_json)
    account_display = consistency["account_number_display"]

    assert account_display["missing_bureaus"] == ["experian", "transunion"]
    assert account_display["normalized"]["transunion"] is None
    assert account_display["normalized"]["experian"] is None
    assert account_display["normalized"]["equifax"]["last4"] == "1234"
    assert account_display["normalized"]["equifax"]["mask_class"] == "masked_last4"


def test_account_number_display_groups_by_last4_and_mask() -> None:
    bureaus_json = {
        "transunion": {"account_number_display": "****1234"},
        "experian": {"account_number_display": "XXXX-1234"},
        "equifax": {"account_number_display": "***1234"},
    }

    consistency = compute_field_consistency(bureaus_json)
    account_display = consistency["account_number_display"]

    assert account_display["consensus"] == "unanimous"
    assert account_display["disagreeing_bureaus"] == []
    mask_classes = {
        account_display["normalized"][bureau]["mask_class"]
        for bureau in ("transunion", "experian", "equifax")
    }
    assert mask_classes == {"masked_last4"}


def test_account_number_display_full_number_matches_masked() -> None:
    bureaus_json = {
        "transunion": {"account_number_display": "123456781234"},
        "experian": {"account_number_display": "****1234"},
        "equifax": {},
    }

    consistency = compute_field_consistency(bureaus_json)
    account_display = consistency["account_number_display"]

    assert account_display["consensus"] == "majority"
    assert account_display["disagreeing_bureaus"] == ["equifax"]


def test_account_number_display_detects_digit_conflict() -> None:
    bureaus_json = {
        "transunion": {"account_number_display": "1111222233334444"},
        "experian": {"account_number_display": "0000222233334444"},
        "equifax": {},
    }

    consistency = compute_field_consistency(bureaus_json)
    account_display = consistency["account_number_display"]

    assert account_display["consensus"] == "split"
    assert {"experian", "transunion"}.issubset(set(account_display["disagreeing_bureaus"]))


def test_account_number_display_detects_alphabetic_conflict() -> None:
    bureaus_json = {
        "transunion": {"account_number_display": "ACCOUNT ABC"},
        "experian": {"account_number_display": "ACCOUNT XYZ"},
        "equifax": {},
    }

    consistency = compute_field_consistency(bureaus_json)
    account_display = consistency["account_number_display"]

    assert account_display["consensus"] == "split"
    assert {"experian", "transunion"}.issubset(set(account_display["disagreeing_bureaus"]))


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


def test_amount_tolerance_within_threshold_is_unanimous() -> None:
    bureaus_json = {
        "transunion": {"balance_owed": "$1,000.00"},
        "experian": {"balance_owed": "1000.40"},
        "equifax": {"balance_owed": "1001"},
    }

    consistency = compute_field_consistency(bureaus_json)

    assert consistency["balance_owed"]["consensus"] == "unanimous"
    assert compute_inconsistent_fields(bureaus_json) == {}


def test_date_tolerance_allows_small_differences_for_last_payment() -> None:
    bureaus_json = {
        "transunion": {"last_payment": "2024-01-01"},
        "experian": {"last_payment": "01/04/2024"},
        "equifax": {"last_payment": "2024-01-02"},
    }

    consistency = compute_field_consistency(bureaus_json)

    assert consistency["last_payment"]["consensus"] == "unanimous"
    assert compute_inconsistent_fields(bureaus_json) == {}


def test_last_verified_within_tolerance_is_unanimous() -> None:
    bureaus_json = {
        "transunion": {"last_verified": "2024-03-01"},
        "experian": {"last_verified": "03/04/2024"},
        "equifax": {"last_verified": "2024-03-06"},
    }

    consistency = compute_field_consistency(bureaus_json)

    assert consistency["last_verified"]["consensus"] == "unanimous"
    assert compute_inconsistent_fields(bureaus_json) == {}


def test_last_verified_beyond_tolerance_detected_as_mismatch() -> None:
    bureaus_json = {
        "transunion": {"last_verified": "2024-03-01"},
        "experian": {"last_verified": "03/08/2024"},
        "equifax": {"last_verified": "2024-03-01"},
    }

    consistency = compute_field_consistency(bureaus_json)

    assert consistency["last_verified"]["consensus"] == "majority"
    inconsistencies = compute_inconsistent_fields(bureaus_json)
    assert "last_verified" in inconsistencies


def test_last_verified_normalizes_with_mdy_convention() -> None:
    bureaus_json = {
        "transunion": {"last_verified": "04/05/2024"},
    }

    with override_validation_context({"convention": "MDY", "month_language": "en"}):
        consistency = compute_field_consistency(bureaus_json)

    normalized = consistency["last_verified"]["normalized"]
    assert normalized["transunion"] == "2024-04-05"


def test_last_verified_normalizes_with_dmy_convention() -> None:
    bureaus_json = {
        "transunion": {"last_verified": "04/05/2024"},
    }

    with override_validation_context({"convention": "DMY", "month_language": "en"}):
        consistency = compute_field_consistency(bureaus_json)

    normalized = consistency["last_verified"]["normalized"]
    assert normalized["transunion"] == "2024-05-04"


def test_normalize_date_respects_dmy_convention() -> None:
    bureaus_json = {
        "transunion": {"date_opened": "1/2/23"},
        "experian": {"date_opened": "1/2/23"},
        "equifax": {"date_opened": "1/2/23"},
    }

    with override_validation_context({"convention": "DMY", "month_language": "en"}):
        consistency = compute_field_consistency(bureaus_json)

    normalized = consistency["date_opened"]["normalized"]
    assert normalized["transunion"] == "2023-02-01"


def test_normalize_date_handles_hebrew_months() -> None:
    bureaus_json = {
        "transunion": {"date_opened": "15 אוג׳ 2022"},
        "experian": {"date_opened": "15 אוג׳ 2022"},
        "equifax": {"date_opened": "15 אוג׳ 2022"},
    }

    with override_validation_context({"convention": "DMY", "month_language": "he"}):
        consistency = compute_field_consistency(bureaus_json)

    normalized = consistency["date_opened"]["normalized"]
    assert normalized["equifax"] == "2022-08-15"


def test_payment_frequency_is_canonicalized() -> None:
    bureaus_json = {
        "transunion": {"payment_frequency": "Monthly (every month)"},
        "experian": {"payment_frequency": "monthly"},
        "equifax": {"payment_frequency": "Once a month"},
    }

    consistency = compute_field_consistency(bureaus_json)

    normalized = consistency["payment_frequency"]["normalized"]
    assert {normalized["transunion"], normalized["experian"], normalized["equifax"]} == {
        "monthly"
    }
    assert consistency["payment_frequency"]["consensus"] == "unanimous"


def test_term_length_normalizes_years_to_months() -> None:
    bureaus_json = {
        "transunion": {"term_length": "30 years"},
        "experian": {"term_length": "360 Month(s)"},
        "equifax": {"term_length": 360},
    }

    consistency = compute_field_consistency(bureaus_json)

    normalized = consistency["term_length"]["normalized"]
    assert normalized["transunion"] == 360
    assert normalized["experian"] == 360
    assert normalized["equifax"] == 360
    assert consistency["term_length"]["consensus"] == "unanimous"
