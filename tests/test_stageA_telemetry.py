import json
import re

import pytest

import backend.config as config
from backend.core.case_store import api as cs_api, telemetry
from backend.core.logic.report_analysis import problem_detection as pd
from backend.core import orchestrators as orch
from backend.core.ai.models import AIAdjudicateResponse

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b")
SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
LONG_DIGIT_RE = re.compile(r"\b\d{8,}\b")
ADDR_RE = re.compile(r"\b(street|road|avenue|drive|st|rd|ave|dr)\b", re.I)

def _assert_no_pii(events):
    payload = json.dumps([fields for _, fields in events])
    for pat in (EMAIL_RE, PHONE_RE, SSN_RE, LONG_DIGIT_RE, ADDR_RE):
        assert not pat.search(payload)

def _setup_case(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "CASESTORE_DIR", str(tmp_path))
    session_id = "sess"
    case = cs_api.create_session_case(session_id)
    cs_api.save_session_case(case)
    base = {
        "balance_owed": 100.0,
        "credit_limit": 1000.0,
        "payment_status": "",
        "account_status": "",
        "two_year_payment_history": "",
        "days_late_7y": "",
    }
    return session_id, base

def test_rules_only_telemetry(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ENABLE_CASESTORE_STAGEA", True)
    monkeypatch.setattr(config, "ENABLE_AI_ADJUDICATOR", False)
    session_id, base = _setup_case(monkeypatch, tmp_path)
    cs_api.upsert_account_fields(session_id, "acc_clean", "Experian", dict(base))
    cs_api.upsert_account_fields(
        session_id, "acc_bad", "Experian", dict(base, past_due_amount=125.0)
    )
    events: list[tuple[str, dict]] = []
    telemetry.set_emitter(lambda e, f: events.append((e, f)))
    pd.run_stage_a(session_id, [])
    orch.collect_stageA_problem_accounts(session_id)
    telemetry.set_emitter(None)
    eval_events = [f for e, f in events if e == "stageA_eval"]
    assert len(eval_events) == 2
    for ev in eval_events:
        assert ev["decision_source"] == "rules"
        assert ev["tier"] == "none"
        assert ev["confidence"] == 0.0
    orch_events = [f for e, f in events if e == "stageA_orchestrated"]
    assert len(orch_events) == 1
    assert orch_events[0]["included"] is True
    fallback_events = [f for e, f in events if e == "stageA_fallback"]
    assert not fallback_events
    _assert_no_pii(events)

def test_ai_adopted_telemetry(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ENABLE_CASESTORE_STAGEA", True)
    monkeypatch.setattr(config, "ENABLE_AI_ADJUDICATOR", True)
    session_id, base = _setup_case(monkeypatch, tmp_path)
    cs_api.upsert_account_fields(session_id, "acc1", "Experian", dict(base))
    events: list[tuple[str, dict]] = []
    telemetry.set_emitter(lambda e, f: events.append((e, f)))
    resp = AIAdjudicateResponse(
        primary_issue="collection",
        tier="Tier1",
        confidence=0.9,
        problem_reasons=["ai_reason"],
        fields_used=["balance_owed"],
    )
    def fake_call(session, req):
        telemetry.emit(
            "stageA_ai_call", attempt=1, status="ok", duration_ms=5.0, confidence=0.9
        )
        return resp
    monkeypatch.setattr(pd, "call_adjudicator", fake_call)
    pd.run_stage_a(session_id, [])
    orch.collect_stageA_problem_accounts(session_id)
    telemetry.set_emitter(None)
    eval_events = [f for e, f in events if e == "stageA_eval"]
    assert len(eval_events) == 1
    ev = eval_events[0]
    assert ev["decision_source"] == "ai"
    assert ev["tier"] == "Tier1"
    assert ev["primary_issue"] == "collection"
    assert ev["ai_latency_ms"] and ev["ai_latency_ms"] > 0
    assert not [f for e, f in events if e == "stageA_fallback"]
    orch_events = [f for e, f in events if e == "stageA_orchestrated"]
    assert len(orch_events) == 1
    assert orch_events[0]["decision_source"] == "ai"
    assert orch_events[0]["included"] is True
    _assert_no_pii(events)

def test_ai_fallback_low_confidence(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ENABLE_CASESTORE_STAGEA", True)
    monkeypatch.setattr(config, "ENABLE_AI_ADJUDICATOR", True)
    session_id, base = _setup_case(monkeypatch, tmp_path)
    cs_api.upsert_account_fields(
        session_id, "acc1", "Experian", dict(base, past_due_amount=200.0)
    )
    events: list[tuple[str, dict]] = []
    telemetry.set_emitter(lambda e, f: events.append((e, f)))
    resp = AIAdjudicateResponse(
        primary_issue="collection",
        tier="Tier1",
        confidence=0.4,
        problem_reasons=["ai_reason"],
        fields_used=["balance_owed"],
    )
    def fake_call(session, req):
        telemetry.emit(
            "stageA_ai_call", attempt=1, status="ok", duration_ms=6.0, confidence=0.4
        )
        return resp
    monkeypatch.setattr(pd, "call_adjudicator", fake_call)
    pd.run_stage_a(session_id, [])
    orch.collect_stageA_problem_accounts(session_id)
    telemetry.set_emitter(None)
    eval_events = [f for e, f in events if e == "stageA_eval"]
    assert eval_events[0]["decision_source"] == "rules"
    fallback = [f for e, f in events if e == "stageA_fallback"]
    assert fallback and fallback[0]["reason"] == "low_confidence"
    assert fallback[0]["ai_confidence"] == pytest.approx(0.4)
    orch_events = [f for e, f in events if e == "stageA_orchestrated"]
    assert orch_events[0]["decision_source"] == "rules"
    assert orch_events[0]["included"] is True
    _assert_no_pii(events)

def test_ai_fallback_schema_reject(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ENABLE_CASESTORE_STAGEA", True)
    monkeypatch.setattr(config, "ENABLE_AI_ADJUDICATOR", True)
    session_id, base = _setup_case(monkeypatch, tmp_path)
    cs_api.upsert_account_fields(
        session_id, "acc1", "Experian", dict(base, past_due_amount=200.0)
    )
    events: list[tuple[str, dict]] = []
    telemetry.set_emitter(lambda e, f: events.append((e, f)))
    def fake_call(session, req):
        telemetry.emit(
            "stageA_ai_call", attempt=1, status="ValidationError", duration_ms=7.0
        )
        return None
    monkeypatch.setattr(pd, "call_adjudicator", fake_call)
    pd.run_stage_a(session_id, [])
    orch.collect_stageA_problem_accounts(session_id)
    telemetry.set_emitter(None)
    eval_events = [f for e, f in events if e == "stageA_eval"]
    assert eval_events[0]["decision_source"] == "rules"
    fallback = [f for e, f in events if e == "stageA_fallback"]
    assert fallback and fallback[0]["reason"] == "schema_reject"
    orch_events = [f for e, f in events if e == "stageA_orchestrated"]
    assert orch_events[0]["included"] is True
    _assert_no_pii(events)

def test_ai_fallback_timeout(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ENABLE_CASESTORE_STAGEA", True)
    monkeypatch.setattr(config, "ENABLE_AI_ADJUDICATOR", True)
    session_id, base = _setup_case(monkeypatch, tmp_path)
    cs_api.upsert_account_fields(
        session_id, "acc1", "Experian", dict(base, past_due_amount=200.0)
    )
    events: list[tuple[str, dict]] = []
    telemetry.set_emitter(lambda e, f: events.append((e, f)))
    def fake_call(session, req):
        telemetry.emit(
            "stageA_ai_call", attempt=1, status="TimeoutException", duration_ms=8.0
        )
        return None
    monkeypatch.setattr(pd, "call_adjudicator", fake_call)
    pd.run_stage_a(session_id, [])
    orch.collect_stageA_problem_accounts(session_id)
    telemetry.set_emitter(None)
    eval_events = [f for e, f in events if e == "stageA_eval"]
    assert eval_events[0]["decision_source"] == "rules"
    fallback = [f for e, f in events if e == "stageA_fallback"]
    assert fallback and fallback[0]["reason"] == "timeout"
    orch_events = [f for e, f in events if e == "stageA_orchestrated"]
    assert orch_events[0]["included"] is True
    _assert_no_pii(events)
