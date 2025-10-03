from backend.core.ai import eligibility_policy as policy
from backend.core.ai.report_compare import classify_reporting_pattern


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
