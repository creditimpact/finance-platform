import pytest
from backend.core.orchestrators import resolve_cross_bureau


def test_tier_precedence_and_ai_source():
    decisions = [
        {
            "primary_issue": "severe_delinquency",
            "tier": "Tier2",
            "confidence": 0.8,
            "problem_reasons": ["late payment"],
            "decision_source": "rules",
        },
        {
            "primary_issue": "collection",
            "tier": "Tier1",
            "confidence": 0.7,
            "problem_reasons": ["collection account"],
            "decision_source": "ai",
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
