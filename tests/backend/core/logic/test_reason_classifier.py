import pytest

from backend.core.logic.reason_classifier import classify_reason, decide_send_to_ai


def test_classify_all_missing():
    result = classify_reason({
        "experian": None,
        "equifax": "",
        "transunion": "--",
    })

    assert result["reason_code"] == "C6_ALL_MISSING"
    assert result["missing_count"] == 3
    assert result["present_count"] == 0
    assert result["distinct_values"] == 0
    assert result["is_missing"] is True
    assert result["is_mismatch"] is False


def test_classify_only_one_present():
    result = classify_reason({
        "experian": "open",
        "equifax": None,
        "transunion": "",
    })

    assert result["reason_code"] == "C2_ONE_MISSING"
    assert result["missing_count"] == 2
    assert result["present_count"] == 1
    assert result["distinct_values"] == 1
    assert result["is_missing"] is True
    assert result["is_mismatch"] is False


def test_classify_two_present_agree_one_missing():
    result = classify_reason({
        "experian": "closed",
        "equifax": "closed",
        "transunion": None,
    })

    assert result["reason_code"] == "C1_TWO_PRESENT_ONE_MISSING"
    assert result["missing_count"] == 1
    assert result["present_count"] == 2
    assert result["distinct_values"] == 1
    assert result["is_missing"] is True
    assert result["is_mismatch"] is False


def test_classify_two_present_conflict_one_missing():
    result = classify_reason({
        "experian": "open",
        "equifax": "closed",
        "transunion": None,
    })

    assert result["reason_code"] == "C3_TWO_PRESENT_CONFLICT"
    assert result["missing_count"] == 1
    assert result["present_count"] == 2
    assert result["distinct_values"] == 2
    assert result["is_missing"] is True
    assert result["is_mismatch"] is True


def test_classify_two_match_one_diff():
    result = classify_reason({
        "experian": "open",
        "equifax": "open",
        "transunion": "closed",
    })

    assert result["reason_code"] == "C4_TWO_MATCH_ONE_DIFF"
    assert result["missing_count"] == 0
    assert result["present_count"] == 3
    assert result["distinct_values"] == 2
    assert result["is_missing"] is False
    assert result["is_mismatch"] is True


def test_classify_all_diff():
    result = classify_reason({
        "experian": "open",
        "equifax": "closed",
        "transunion": "charged off",
    })

    assert result["reason_code"] == "C5_ALL_DIFF"
    assert result["missing_count"] == 0
    assert result["present_count"] == 3
    assert result["distinct_values"] == 3
    assert result["is_missing"] is False
    assert result["is_mismatch"] is True


def test_normalizes_whitespace_and_missing_markers():
    result = classify_reason({
        "experian": "  open  ",
        "equifax": " -- ",
        "transunion": " ",
    })

    assert result["reason_code"] == "C2_ONE_MISSING"
    assert result["missing_count"] == 2
    assert result["present_count"] == 1
    assert result["distinct_values"] == 1
    assert result["is_missing"] is True
    assert result["is_mismatch"] is False


@pytest.mark.parametrize(
    "field, reason, expected",
    [
        ("account_type", "C3_TWO_PRESENT_CONFLICT", True),
        ("account_rating", {"reason_code": "C4_TWO_MATCH_ONE_DIFF"}, True),
        ("creditor_type", "C5_ALL_DIFF", True),
        ("account_type", "C1_TWO_PRESENT_ONE_MISSING", False),
        ("creditor_type", "C2_ONE_MISSING", False),
        ("balance", "C3_TWO_PRESENT_CONFLICT", False),
        ("account_rating", {}, False),
    ],
)
def test_decide_send_to_ai(field, reason, expected):
    assert decide_send_to_ai(field, reason) is expected

