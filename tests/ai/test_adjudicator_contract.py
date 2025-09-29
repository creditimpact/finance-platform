"""Contract tests for AI adjudicator normalization."""
from __future__ import annotations

from backend.core.ai.adjudicator import _normalize_and_validate_decision


def test_normalizes_decision_to_match_flags() -> None:
    """Flags win over the raw decision when they disagree."""

    payload = {
        "decision": "MERGE",
        "reason": "  inconsistent debt  ",
        "flags": {"account_match": "true", "debt_match": "FALSE"},
    }

    normalized, was_normalized = _normalize_and_validate_decision(payload)

    assert normalized["decision"] == "same_account_diff_debt"
    assert normalized["reason"] == "inconsistent debt"
    assert normalized["flags"] == {"account_match": True, "debt_match": False}
    assert normalized["normalized"] is True
    assert was_normalized is True


def test_keeps_merge_when_flags_strong_match() -> None:
    """The adjudicator keeps a merge decision when both flags are true."""

    payload = {
        "decision": " merge ",
        "reason": "Shared account number and balance",
        "flags": {"account_match": True, "debt_match": True},
    }

    normalized, was_normalized = _normalize_and_validate_decision(payload)

    assert normalized["decision"] == "same_account_same_debt"
    assert normalized["reason"] == "Shared account number and balance"
    assert normalized["flags"] == {"account_match": True, "debt_match": True}
    assert normalized["normalized"] is True
    assert was_normalized is True
