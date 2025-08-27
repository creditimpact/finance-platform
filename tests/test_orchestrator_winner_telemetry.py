import pytest

import backend.config as config
from backend.core.case_store import api as cs_api, telemetry
from backend.core import orchestrators as orch


def _run(decisions, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CASESTORE_DIR", str(tmp_path))
    monkeypatch.setattr(config, "ENABLE_CROSS_BUREAU_RESOLUTION", True)
    monkeypatch.setattr(config, "ENABLE_AI_ADJUDICATOR", True)
    session_id = "sess"
    case = cs_api.create_session_case(session_id)
    cs_api.save_session_case(case)
    base_fields = {
        "account_number": "00001234",
        "creditor_type": "bank",
        "date_opened": "2020-01-01",
    }
    for idx, dec in enumerate(decisions, 1):
        acc_id = f"a{idx}"
        bureau = dec["bureau"]
        payload = dict(dec)
        payload.pop("bureau", None)
        cs_api.upsert_account_fields(session_id, acc_id, bureau, base_fields)
        cs_api.append_artifact(session_id, acc_id, "stageA_detection", payload)
    events: list[tuple[str, dict]] = []
    telemetry.set_emitter(lambda e, f: events.append((e, f)))
    resolved = orch.collect_stageA_logical_accounts(session_id)
    telemetry.set_emitter(None)
    return resolved, events


def test_tier_precedence_winner_telemetry(tmp_path, monkeypatch):
    decisions = [
        {
            "bureau": "Experian",
            "primary_issue": "late_payment",
            "tier": "Tier2",
            "confidence": 0.5,
            "problem_reasons": ["late"],
            "decision_source": "rules",
        },
        {
            "bureau": "TransUnion",
            "primary_issue": "collection",
            "tier": "Tier1",
            "confidence": 0.9,
            "problem_reasons": ["collection"],
            "decision_source": "ai",
        },
    ]
    resolved, events = _run(decisions, tmp_path, monkeypatch)
    assert len(resolved) == 1
    winner_events = [f for e, f in events if e == "stageA_cross_bureau_winner"]
    assert len(winner_events) == 1
    event = winner_events[0]
    assert event["tier"] == "Tier1"
    assert event["winner_bureau"] == "TransUnion"
    assert event["reasons_count"] == 2
    assert "problem_reasons" not in event
    assert event["members"] == 2


def test_confidence_tiebreak_winner_telemetry(tmp_path, monkeypatch):
    decisions = [
        {
            "bureau": "Experian",
            "primary_issue": "collection",
            "tier": "Tier1",
            "confidence": 0.4,
            "problem_reasons": ["r1"],
            "decision_source": "rules",
        },
        {
            "bureau": "TransUnion",
            "primary_issue": "collection",
            "tier": "Tier1",
            "confidence": 0.9,
            "problem_reasons": ["r2"],
            "decision_source": "rules",
        },
    ]
    resolved, events = _run(decisions, tmp_path, monkeypatch)
    assert len(resolved) == 1
    winner = [f for e, f in events if e == "stageA_cross_bureau_winner"][0]
    assert winner["winner_bureau"] == "TransUnion"
    assert winner["confidence"] == pytest.approx(0.9)
