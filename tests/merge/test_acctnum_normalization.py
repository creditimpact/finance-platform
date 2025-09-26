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


def test_account_number_level_visible_digits_match() -> None:
    assert (
        account_merge.account_number_level("349992*****", "3499921234567")
        == "exact_or_known_match"
    )


def test_account_number_level_requires_substring() -> None:
    assert (
        account_merge.account_number_level("555550*****", "555555*****") == "none"
    )


def test_acctnum_match_level_debug_payload() -> None:
    level, debug = account_merge.acctnum_match_level("****6789", "123456789")
    assert level == "exact_or_known_match"
    assert debug["short"] == "6789"
    assert debug["long"] == "123456789"
    assert debug["a"]["digits"] == "6789"
    assert debug["b"]["digits"] == "123456789"


def test_acctnum_match_level_requires_visible_digits() -> None:
    level, debug = account_merge.acctnum_match_level("********", "XXXX")
    assert level == "none"
    assert debug.get("why") == "empty"
