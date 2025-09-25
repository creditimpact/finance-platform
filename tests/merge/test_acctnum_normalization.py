"""Tests for account-number normalization and match levels."""
from __future__ import annotations

from backend.core.logic.report_analysis import account_merge


def test_normalize_acctnum_collapses_masks() -> None:
    result = account_merge.normalize_acctnum("**** ****1234")
    assert result["digits"] == "1234"
    assert result["digits_last4"] == "1234"
    assert result["digits_last6"] is None
    assert result["canon_mask"] == "*1234"
    assert result["has_digits"] is True


def test_normalize_acctnum_handles_mask_only() -> None:
    result = account_merge.normalize_acctnum("****")
    assert result["digits"] == ""
    assert result["digits_last4"] is None
    assert result["digits_last6"] is None
    assert result["canon_mask"] == "*"
    assert result["has_digits"] is False


def test_account_number_level_exact() -> None:
    assert account_merge.account_number_level("1234567890", "1234567890") == "exact"


def test_account_number_level_last6() -> None:
    assert account_merge.account_number_level("99123456", "123456") == "last6"


def test_account_number_level_last4() -> None:
    assert account_merge.account_number_level("12345678", "005678") == "last4"


def test_acctnum_match_level_masked_exact() -> None:
    level, debug = account_merge.acctnum_match_level("**12**34", "**12**34")
    assert level == "exact"
    assert debug["left"]["canon_mask"] == "*12*34"
    assert debug["right"]["canon_mask"] == "*12*34"


def test_acctnum_match_level_prefers_last6() -> None:
    level, _ = account_merge.acctnum_match_level("99123456", "123456")
    assert level == "last6"
