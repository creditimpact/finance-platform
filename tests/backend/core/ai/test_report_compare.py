from backend.core.ai import eligibility_policy as policy
import pytest

from backend.core.ai.report_compare import classify_reporting_pattern, compute_reason_flags


def test_case_1_single_present():
    values = {"equifax": "value", "experian": None, "transunion": ""}
    assert classify_reporting_pattern(values) == "case_1"


def test_case_2_one_missing_two_match():
    values = {"equifax": "VALUE", "experian": " value ", "transunion": None}
    assert classify_reporting_pattern(values) == "case_2"


def test_case_3_one_missing_two_different():
    values = {"equifax": "foo", "experian": "bar", "transunion": "--"}
    assert classify_reporting_pattern(values) == "case_3"


def test_case_4_all_present_two_match():
    values = {"equifax": "foo", "experian": "foo", "transunion": "bar"}
    assert classify_reporting_pattern(values) == "case_4"


def test_case_5_all_present_all_different():
    values = {"equifax": "foo", "experian": "bar", "transunion": "baz"}
    assert classify_reporting_pattern(values) == "case_5"


def test_case_6_all_missing():
    values = {"equifax": None, "experian": "--", "transunion": "!!!"}
    assert classify_reporting_pattern(values) == "case_6"


def test_canonicalization_with_history():
    values = {
        "equifax": [" OK ", "LATE"],
        "experian": ["ok", "late"],
        "transunion": None,
    }
    assert policy.canonicalize_history(values["equifax"]) == "ok|late"
    assert classify_reporting_pattern(values) == "case_2"


@pytest.mark.parametrize(
    "pattern,expected",
    [
        ("case_1", {"missing": True, "mismatch": False, "both": False, "eligible": True}),
        ("case_2", {"missing": True, "mismatch": False, "both": False, "eligible": True}),
        ("case_3", {"missing": True, "mismatch": True, "both": True, "eligible": True}),
        ("case_4", {"missing": False, "mismatch": True, "both": False, "eligible": True}),
        ("case_5", {"missing": False, "mismatch": True, "both": False, "eligible": True}),
        ("case_6", {"missing": True, "mismatch": False, "both": False, "eligible": True}),
    ],
)
def test_reason_flags_for_always_eligible(pattern, expected):
    result = compute_reason_flags("date_opened", pattern, match_matrix={})
    assert result == expected


@pytest.mark.parametrize(
    "pattern,expected",
    [
        ("case_1", {"missing": True, "mismatch": False, "both": False, "eligible": False}),
        ("case_2", {"missing": True, "mismatch": False, "both": False, "eligible": False}),
        ("case_3", {"missing": True, "mismatch": True, "both": True, "eligible": True}),
        ("case_4", {"missing": False, "mismatch": True, "both": False, "eligible": True}),
        ("case_5", {"missing": False, "mismatch": True, "both": False, "eligible": True}),
        ("case_6", {"missing": True, "mismatch": False, "both": False, "eligible": False}),
    ],
)
def test_reason_flags_for_conditional(pattern, expected):
    result = compute_reason_flags("account_number_display", pattern, match_matrix={})
    assert result == expected
