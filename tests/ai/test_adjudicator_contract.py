"""Contract tests for AI adjudicator normalization."""
from __future__ import annotations

import pytest

from backend.core.ai.adjudicator import (
    AdjudicatorError,
    _normalize_and_validate_decision,
    validate_ai_payload,
)


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


def test_normalizes_duplicate_contract() -> None:
    payload = {
        "decision": "duplicate",
        "reason": "A cites B as original creditor",
        "flags": {"duplicate": True},
    }

    normalized, was_normalized = _normalize_and_validate_decision(payload)

    assert normalized["decision"] == "duplicate"
    assert normalized["reason"] == "A cites B as original creditor"
    assert normalized["flags"] == {"duplicate": True}
    assert "normalized" not in normalized
    assert was_normalized is False


def test_normalizes_not_duplicate_contract() -> None:
    payload = {
        "decision": "not_duplicate",
        "reason": "different creditors",
        "flags": {"duplicate": False},
    }

    normalized, was_normalized = _normalize_and_validate_decision(payload)

    assert normalized["decision"] == "not_duplicate"
    assert normalized["flags"] == {"duplicate": False}
    assert "normalized" not in normalized
    assert was_normalized is False


def test_validate_ai_payload_coerces_boolean_flags() -> None:
    payload = {
        "decision": "same_account_same_debt",
        "reason": "accounts align",
        "flags": {"account_match": True, "debt_match": False},
    }

    normalized = validate_ai_payload(payload)

    assert normalized == {
        "decision": "same_account_same_debt",
        "flags": {"account_match": "true", "debt_match": "false"},
        "reason": "accounts align",
    }


def test_validate_ai_payload_defaults_unknown_for_missing_flags() -> None:
    payload = {"decision": "same_account_same_debt", "reason": "accounts align"}

    normalized = validate_ai_payload(payload)

    assert normalized["flags"] == {
        "account_match": "unknown",
        "debt_match": "unknown",
    }


def test_validate_ai_payload_rejects_invalid_flag() -> None:
    payload = {
        "decision": "same_account_same_debt",
        "flags": {"account_match": "maybe", "debt_match": "false"},
    }

    with pytest.raises(AdjudicatorError, match="Flags outside contract"):
        validate_ai_payload(payload)


def test_validate_ai_payload_accepts_duplicate_contract() -> None:
    payload = {
        "decision": "duplicate",
        "reason": "oc matches",
        "flags": {"duplicate": True},
    }

    normalized = validate_ai_payload(payload)

    assert normalized == {
        "decision": "duplicate",
        "flags": {"duplicate": True},
        "reason": "oc matches",
    }
