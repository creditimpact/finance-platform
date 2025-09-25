"""Tests for account-number tokenization and match levels."""
from __future__ import annotations

import pytest

from backend.core.logic.report_analysis import account_merge
from backend.core.logic.report_analysis.account_merge import AcctnumToken


def test_account_number_level_exact() -> None:
    assert account_merge.account_number_level("001234567890", "1234567890") == "exact"


def test_account_number_level_last4() -> None:
    assert account_merge.account_number_level("12345678", "005678") == "last4"


def test_account_number_level_masked_match(monkeypatch: pytest.MonkeyPatch) -> None:
    token_a = AcctnumToken(value="012", masked_value="**012", signature="MMDDD")
    token_b = AcctnumToken(value="012", masked_value="xx012", signature="MMDDD")

    def _fake_normalize(raw: str) -> dict[str, set[str | AcctnumToken]]:
        if raw == "first":
            return {"last4": set(), "last5": set(), "tokens": {token_a}}
        if raw == "second":
            return {"last4": set(), "last5": set(), "tokens": {token_b}}
        raise AssertionError(f"Unexpected raw value: {raw}")

    monkeypatch.setattr(account_merge, "normalize_acctnum", _fake_normalize)

    assert account_merge.account_number_level("first", "second") == "masked_match"
