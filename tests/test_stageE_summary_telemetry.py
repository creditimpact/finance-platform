import pytest

import backend.config as config
from backend.core.case_store import api as cs_api, telemetry
from backend.core import orchestrators as orch
from backend.core.telemetry import stageE_summary


def _setup(monkeypatch, tmp_path, accounts):
    """Create session with given account decision sources.

    accounts: list of tuples (acc_id, decision_source)
    """
    monkeypatch.setattr(config, "CASESTORE_DIR", str(tmp_path))
    monkeypatch.setattr(config, "ENABLE_CROSS_BUREAU_RESOLUTION", False)
    monkeypatch.setattr(config, "ENABLE_AI_ADJUDICATOR", True)
    session_id = "sess"
    case = cs_api.create_session_case(session_id)
    cs_api.save_session_case(case)
    base_fields = {
        "account_number": "00001234",
        "creditor_type": "bank",
        "date_opened": "2020-01-01",
    }
    for acc_id, source in accounts:
        cs_api.upsert_account_fields(session_id, acc_id, "Experian", base_fields)
        payload = {
            "primary_issue": "late_payment",
            "tier": "Tier1",
            "confidence": 0.9,
            "problem_reasons": ["r"],
            "decision_source": source,
        }
        cs_api.append_artifact(session_id, acc_id, "stageA_detection", payload)
    return session_id


def test_rules_only_adoption_zero(tmp_path, monkeypatch):
    session_id = _setup(monkeypatch, tmp_path, [("a1", "rules"), ("a2", "rules")])
    events = []
    telemetry.set_emitter(lambda e, f: events.append((e, f)))
    orch.collect_stageA_logical_accounts(session_id)
    telemetry.set_emitter(None)
    summary = [f for e, f in events if e == "stageE_summary"][0]
    assert summary["ai_adoption_pct"] == 0.0
    assert summary["fallback_pct"] == 0.0
    assert summary["problem_accounts"] == 2


def test_half_ai_half_fallback(tmp_path, monkeypatch):
    session_id = _setup(monkeypatch, tmp_path, [("a1", "ai"), ("a2", "rules")])
    stageE_summary.record_stageA_event(
        "stageA_fallback", {"session_id": session_id, "account_id": "a2", "latency_ms": 100}
    )
    events = []
    telemetry.set_emitter(lambda e, f: events.append((e, f)))
    orch.collect_stageA_logical_accounts(session_id)
    telemetry.set_emitter(None)
    summary = [f for e, f in events if e == "stageE_summary"][0]
    assert summary["ai_adoption_pct"] == pytest.approx(0.5)
    assert summary["fallback_pct"] == pytest.approx(0.5)


def test_latency_p95(tmp_path, monkeypatch):
    session_id = _setup(monkeypatch, tmp_path, [("a1", "ai")])
    stageE_summary.record_stageA_event(
        "stageA_eval", {"session_id": session_id, "ai_latency_ms": 50}
    )
    stageE_summary.record_stageA_event(
        "stageA_eval", {"session_id": session_id, "ai_latency_ms": 100}
    )
    stageE_summary.record_stageA_event(
        "stageA_fallback", {"session_id": session_id, "account_id": "a1", "latency_ms": 300}
    )
    events = []
    telemetry.set_emitter(lambda e, f: events.append((e, f)))
    orch.collect_stageA_logical_accounts(session_id)
    telemetry.set_emitter(None)
    summary = [f for e, f in events if e == "stageE_summary"][0]
    assert summary["ai_latency_p95_ms"] == pytest.approx(300)


def test_emitted_with_no_problem_accounts(tmp_path, monkeypatch):
    session_id = _setup(monkeypatch, tmp_path, [])
    events = []
    telemetry.set_emitter(lambda e, f: events.append((e, f)))
    orch.collect_stageA_logical_accounts(session_id)
    telemetry.set_emitter(None)
    summary_events = [f for e, f in events if e == "stageE_summary"]
    assert len(summary_events) == 1
    summary = summary_events[0]
    assert summary["problem_accounts"] == 0
    allowed = {
        "session_id",
        "total_accounts",
        "problem_accounts",
        "ai_adoption_pct",
        "fallback_pct",
        "ai_latency_p95_ms",
        "duration_ms",
    }
    assert set(summary) == allowed
