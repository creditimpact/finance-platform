from datetime import date

import pytest

from backend.core.logic.report_analysis.account_merge import (
    account_number_level,
    account_numbers_match,
    date_equal,
    date_within,
    digits_only,
    get_merge_cfg,
    match_amount_field,
    match_balance_owed,
    match_field_best_of_9,
    match_payment_amount,
    normalize_amount_field,
    normalize_balance_owed,
    normalize_type,
    to_amount,
    to_date,
)


def test_digits_only_requires_real_digits():
    assert digits_only("**XX--") is None
    assert digits_only(" 12 34  ") == "1234"


def test_to_amount_handles_currency_and_missing():
    assert to_amount("$1,234.56") == pytest.approx(1234.56)
    assert to_amount("(75.10)") == pytest.approx(-75.10)
    assert to_amount("--") is None
    assert to_amount(0) == 0.0


def test_amounts_match_with_tolerance():
    val_a = normalize_amount_field("1,000")
    val_b = normalize_amount_field("1,005.00")
    assert match_amount_field(val_a, val_b, tol_abs=5.0, tol_ratio=0.001)

    val_c = normalize_amount_field("1,016")
    assert not match_amount_field(val_a, val_c, tol_abs=5.0, tol_ratio=0.001)


def test_balance_owed_requires_exact_match():
    bal_a = normalize_balance_owed("100.00")
    bal_b = normalize_balance_owed("100")
    bal_c = normalize_balance_owed("100.01")

    assert match_balance_owed(bal_a, bal_b)
    assert not match_balance_owed(bal_a, bal_c)


def test_payment_amount_zero_rule_respected():
    zero_a = normalize_amount_field("0")
    zero_b = normalize_amount_field("0.00")
    assert not match_payment_amount(
        zero_a,
        zero_b,
        tol_abs=0.0,
        tol_ratio=0.0,
        count_zero_payment_match=0,
    )

    assert match_payment_amount(
        zero_a,
        zero_b,
        tol_abs=0.0,
        tol_ratio=0.0,
        count_zero_payment_match=1,
    )


@pytest.mark.parametrize(
    "a,b,expected",
    [
        ("349992*****", "3499921234567", "exact_or_known_match"),
        ("****6789", "123456789", "exact_or_known_match"),
        ("555550*****", "555555*****", "none"),
        ("abcd", "****", "none"),
    ],
)
def test_account_number_levels(a, b, expected):
    assert account_number_level(a, b) == expected


def test_account_numbers_match_thresholds():
    match, level = account_numbers_match(
        "****6789", "123456789", min_level="exact_or_known_match"
    )
    assert match is True
    assert level == "exact_or_known_match"

    match_low, level_low = account_numbers_match("123", "456", min_level="none")
    assert match_low is False
    assert level_low == "none"


def test_to_date_parses_common_formats():
    assert to_date("2023-05-07") == date(2023, 5, 7)
    assert to_date("05/07/2023") == date(2023, 5, 7)
    assert to_date("20230507") == date(2023, 5, 7)
    assert to_date("2023.5.7") == date(2023, 5, 7)
    assert to_date("--") is None


def test_date_comparisons_work():
    day_a = date(2023, 5, 7)
    day_b = date(2023, 5, 10)
    assert date_equal(day_a, day_a)
    assert not date_equal(day_a, day_b)
    assert date_within(day_a, day_b, 3)
    assert not date_within(day_a, day_b, 2)


def test_normalize_type_aliases_credit_card_and_bank():
    assert normalize_type("US BK CACS") == "u s bank"
    assert normalize_type("Credit-Card - Revolving") == "credit card"
    assert normalize_type("--") is None


def test_match_field_best_of_9_cross_bureau_balance():
    cfg = get_merge_cfg()
    account_a = {
        "transunion": {"balance_owed": "100"},
        "experian": {},
        "equifax": {},
    }
    account_b = {
        "transunion": {"balance_owed": "200"},
        "experian": {},
        "equifax": {"balance_owed": "100.00"},
    }

    matched, aux = match_field_best_of_9("balance_owed", account_a, account_b, cfg)

    assert matched is True
    assert aux["best_pair"] == ("transunion", "equifax")
    assert aux["normalized_values"] == (100.0, 100.0)


def test_match_field_best_of_9_unmatched_has_best_pair():
    cfg = get_merge_cfg()
    account_a = {
        "transunion": {"balance_owed": "100"},
        "experian": {},
        "equifax": {},
    }
    account_b = {
        "transunion": {},
        "experian": {"balance_owed": "175"},
        "equifax": {},
    }

    matched, aux = match_field_best_of_9("balance_owed", account_a, account_b, cfg)

    assert matched is False
    assert aux["best_pair"] == ("transunion", "experian")
    assert aux["normalized_values"] == (100.0, 175.0)


def test_match_field_best_of_9_account_number_aux():
    cfg = get_merge_cfg()
    account_a = {
        "experian": {"account_number": "****1234"},
        "transunion": {},
        "equifax": {},
    }
    account_b = {
        "transunion": {},
        "experian": {},
        "equifax": {"account_number": "00001234"},
    }

    matched, aux = match_field_best_of_9("account_number", account_a, account_b, cfg)

    assert matched is True
    assert aux["best_pair"] == ("experian", "equifax")
    assert aux["acctnum_level"] == "exact_or_known_match"
    assert aux["normalized_values"] == ("1234", "00001234")
    assert aux["acctnum_digits_len_a"] == 4
    assert aux["acctnum_digits_len_b"] == 8
    assert aux["matched"] is True


def test_match_field_best_of_9_missing_values_do_not_match():
    cfg = get_merge_cfg()
    account_a = {"transunion": {"high_balance": "--"}}
    account_b = {"equifax": {"high_balance": ""}}

    matched, aux = match_field_best_of_9("high_balance", account_a, account_b, cfg)

    assert matched is False
    assert aux == {}


def test_match_field_best_of_9_last_payment_within_tolerance():
    cfg = get_merge_cfg()
    account_a = {"experian": {"last_payment": "2023-05-01"}}
    account_b = {"transunion": {"last_payment": "2023-05-05"}}

    matched, aux = match_field_best_of_9("last_payment", account_a, account_b, cfg)

    assert matched is True
    assert aux["best_pair"] == ("experian", "transunion")
    assert aux["normalized_values"] == ("2023-05-01", "2023-05-05")

