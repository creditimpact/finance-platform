import pytest
from backend.core.case_store.models import AccountCase, AccountFields, Bureau
from backend.core.orchestrators import (
    compute_logical_account_key,
    resolve_cross_bureau,
)


def test_compute_logical_account_key_deterministic_and_safe():
    case = AccountCase(
        bureau=Bureau.Equifax,
        fields=AccountFields(
            account_number="1234567890",
            creditor_type="bank",
            date_opened="2020-01-02",
        ),
    )
    key1 = compute_logical_account_key(case)
    key2 = compute_logical_account_key(case)
    assert key1 == key2
    assert "7890" not in key1

    case2 = AccountCase(
        bureau=Bureau.Equifax,
        fields=AccountFields(
            account_number="99991234",
            creditor_type="bank",
            date_opened="2020-01-02",
        ),
    )
    key3 = compute_logical_account_key(case2)
    assert key3 != key1


def test_tier_precedence():
    decisions = [
        {
            "primary_issue": "collection",
            "tier": "Tier1",
            "confidence": 0.7,
            "problem_reasons": ["collection account"],
            "decision_source": "ai",
        },
        {
            "primary_issue": "severe_delinquency",
            "tier": "Tier2",
            "confidence": 0.8,
            "problem_reasons": ["late payment"],
            "decision_source": "rules",
        },
        {
            "primary_issue": "utilization",
            "tier": "Tier3",
            "confidence": 0.9,
            "problem_reasons": ["util"],
            "decision_source": "rules",
        },
        {
            "primary_issue": "high_utilization",
            "tier": "Tier4",
            "confidence": 0.9,
            "problem_reasons": ["util"],
            "decision_source": "rules",
        },
        {
            "primary_issue": "unknown",
            "tier": "none",
            "confidence": 0.9,
            "problem_reasons": ["n/a"],
            "decision_source": "rules",
        },
    ]
    resolved = resolve_cross_bureau(decisions)
    assert resolved["tier"] == "Tier1"
    assert resolved["primary_issue"] == "collection"
    assert resolved["decision_source"] == "ai"


def test_confidence_tiebreak():
    decisions = [
        {
            "primary_issue": "collection",
            "tier": "Tier1",
            "confidence": 0.4,
            "problem_reasons": ["r1"],
            "decision_source": "rules",
        },
        {
            "primary_issue": "collection",
            "tier": "Tier1",
            "confidence": 0.9,
            "problem_reasons": ["r2"],
            "decision_source": "rules",
        },
    ]
    resolved = resolve_cross_bureau(decisions)
    assert resolved["confidence"] == 0.9


def test_reasons_merge_dedup_and_cap():
    reasons1 = [f"r{i}" for i in range(8)]
    reasons2 = [f"r{i}" for i in range(5, 15)]  # overlap and exceed cap
    decisions = [
        {
            "primary_issue": "collection",
            "tier": "Tier1",
            "confidence": 0.5,
            "problem_reasons": reasons1,
        },
        {
            "primary_issue": "collection",
            "tier": "Tier1",
            "confidence": 0.6,
            "problem_reasons": reasons2,
        },
    ]
    resolved = resolve_cross_bureau(decisions)
    assert len(resolved["problem_reasons"]) <= 10
    assert set(resolved["problem_reasons"]) <= set(reasons1 + reasons2)


def test_tier4_only():
    decisions = [
        {
            "primary_issue": "high_utilization",
            "tier": "Tier4",
            "confidence": 0.3,
            "problem_reasons": ["utilization"],
        },
        {
            "primary_issue": "high_utilization",
            "tier": "Tier4",
            "confidence": 0.7,
            "problem_reasons": ["utilization other"],
        },
    ]
    resolved = resolve_cross_bureau(decisions)
    assert resolved["tier"] == "Tier4"
    assert resolved["primary_issue"] == "high_utilization"
