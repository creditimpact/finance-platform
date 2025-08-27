import pytest

import backend.config as config
from backend.core.case_store import telemetry
from backend.core.orchestrators import collect_stageA_logical_accounts


def _run(all_accounts, monkeypatch):
    monkeypatch.setattr(config, "ENABLE_CROSS_BUREAU_RESOLUTION", True)
    monkeypatch.setattr(config, "ENABLE_CASESTORE_STAGEA", False)
    events: list[tuple[str, dict]] = []
    telemetry.set_emitter(lambda e, f: events.append((e, f)))
    resolved = collect_stageA_logical_accounts("sess", all_accounts)
    telemetry.set_emitter(None)
    return resolved, events


def test_tier_precedence_winner_telemetry(monkeypatch):
    accounts = [
        {
            "_detector_is_problem": True,
            "account_id": "acct",
            "bureau": "Experian",
            "primary_issue": "late_payment",
            "tier": "Tier2",
            "confidence": 0.5,
            "problem_reasons": ["late"],
            "decision_source": "rules",
        },
        {
            "_detector_is_problem": True,
            "account_id": "acct",
            "bureau": "TransUnion",
            "primary_issue": "collection",
            "tier": "Tier1",
            "confidence": 0.9,
            "problem_reasons": ["collection"],
            "decision_source": "ai",
        },
    ]
    resolved, events = _run(accounts, monkeypatch)
    assert len(resolved) == 1
    winner_events = [f for e, f in events if e == "stageA_cross_bureau_winner"]
    assert len(winner_events) == 1
    event = winner_events[0]
    assert event["tier"] == "Tier1"
    assert event["winner_bureau"] == "TransUnion"
    assert event["reasons_count"] == 2
    assert "problem_reasons" not in event
    assert event["members"] == 2


def test_confidence_tiebreak_winner_telemetry(monkeypatch):
    accounts = [
        {
            "_detector_is_problem": True,
            "account_id": "acct",
            "bureau": "Experian",
            "primary_issue": "collection",
            "tier": "Tier1",
            "confidence": 0.4,
            "problem_reasons": ["r1"],
            "decision_source": "rules",
        },
        {
            "_detector_is_problem": True,
            "account_id": "acct",
            "bureau": "TransUnion",
            "primary_issue": "collection",
            "tier": "Tier1",
            "confidence": 0.9,
            "problem_reasons": ["r2"],
            "decision_source": "rules",
        },
    ]
    resolved, events = _run(accounts, monkeypatch)
    assert len(resolved) == 1
    winner = [f for e, f in events if e == "stageA_cross_bureau_winner"][0]
    assert winner["winner_bureau"] == "TransUnion"
    assert winner["confidence"] == pytest.approx(0.9)


def test_tier4_winner_telemetry_even_when_excluded(monkeypatch):
    accounts = [
        {
            "_detector_is_problem": True,
            "account_id": "acct",
            "bureau": "Experian",
            "primary_issue": "high_utilization",
            "tier": "Tier4",
            "confidence": 0.3,
            "problem_reasons": ["high util"],
            "decision_source": "rules",
        },
        {
            "_detector_is_problem": True,
            "account_id": "acct",
            "bureau": "TransUnion",
            "primary_issue": "high_utilization",
            "tier": "Tier4",
            "confidence": 0.7,
            "problem_reasons": ["utilization"],
            "decision_source": "rules",
        },
    ]
    resolved, events = _run(accounts, monkeypatch)
    assert resolved == []
    winner = [f for e, f in events if e == "stageA_cross_bureau_winner"][0]
    assert winner["tier"] == "Tier4"
    assert "problem_reasons" not in winner

