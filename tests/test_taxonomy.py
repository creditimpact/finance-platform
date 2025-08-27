import re
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from backend.core.taxonomy import (
    issue_to_tier,
    compare_tiers,
    strongest_tier,
    normalize_decision,
)


def contains_pii(text: str) -> bool:
    pii_regex = re.compile(
        r"("  # email
        r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"  # noqa: W605
        r"|\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b"  # phone
        r"|\b\d{3}-\d{2}-\d{4}\b"  # ssn
        r"|\b\d{9,}\b"  # long digit run
        r")"
    )
    return bool(pii_regex.search(text))


def test_issue_to_tier_mapping():
    cases = {
        "bankruptcy": "Tier1",
        "charge_off": "Tier1",
        "collection": "Tier1",
        "repossession": "Tier1",
        "foreclosure": "Tier1",
        "severe_delinquency": "Tier2",
        "moderate_delinquency": "Tier3",
        "derogatory": "Tier3",
        "high_utilization": "Tier4",
        "none": "none",
        "unknown": "none",
    }
    for issue, tier in cases.items():
        assert issue_to_tier(issue) == tier


def test_tier_comparisons_and_strongest():
    assert compare_tiers("Tier1", "Tier2") == "Tier1"
    assert compare_tiers("Tier4", "Tier3") == "Tier3"
    assert compare_tiers("none", "Tier2") == "Tier2"

    assert strongest_tier(["high_utilization"]) == "Tier4"
    assert strongest_tier(["moderate_delinquency", "high_utilization"]) == "Tier3"
    assert strongest_tier(["collection", "high_utilization", "derogatory"]) == "Tier1"
    assert strongest_tier([]) == "none"


def test_normalize_decision_recomputes_and_clamps():
    reasons = [
        "dup",
        "dup",
        "x" * 250,
    ] + [f"r{i}" for i in range(15)]
    decision = {
        "primary_issue": "collection",
        "tier": "Tier3",
        "decision_source": "ai",
        "problem_reasons": reasons,
        "confidence": 1.5,
    }
    norm = normalize_decision(decision)
    assert norm["primary_issue"] == "collection"
    assert norm["tier"] == "Tier1"  # recomputed
    assert norm["decision_source"] == "ai"
    assert norm["confidence"] == 1.0

    # reasons deduped, trimmed and capped
    assert norm["problem_reasons"][0] == "dup"
    assert len(norm["problem_reasons"][1]) == 200
    assert len(norm["problem_reasons"]) == 10
    assert norm["problem_reasons"].count("dup") == 1

    assert not contains_pii(str(norm))


def test_normalize_decision_unknown_and_negative_confidence():
    decision = {
        "primary_issue": "gibberish",
        "tier": "Tier1",
        "decision_source": "rules",
        "problem_reasons": ["ok"],
        "confidence": -0.2,
    }
    norm = normalize_decision(decision)
    assert norm["primary_issue"] == "unknown"
    assert norm["tier"] == "none"
    assert norm["decision_source"] == "rules"
    assert norm["confidence"] == 0.0
    assert not contains_pii(str(norm))
