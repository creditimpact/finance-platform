from __future__ import annotations

import pytest

import backend.config as config
from backend.core.ai.models import AIAdjudicateResponse
from backend.core.case_store import api as cs_api
from backend.core.logic.report_analysis import problem_detection as pd


@pytest.fixture
def session_case(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CASESTORE_DIR", str(tmp_path))
    session_id = "sess1"
    case = cs_api.create_session_case(session_id)
    cs_api.save_session_case(case)
    base = {
        "balance_owed": 100.0,
        "credit_limit": 1000.0,
        "high_balance": 500.0,
        "payment_status": "",
        "account_status": "",
        "two_year_payment_history": "",
        "days_late_7y": "",
    }
    cs_api.upsert_account_fields(
        session_id, "acc1", "Experian", dict(base, past_due_amount=0.0)
    )
    cs_api.upsert_account_fields(
        session_id, "acc2", "Experian", dict(base, past_due_amount=125.0)
    )
    cs_api.upsert_account_fields(
        session_id,
        "acc3",
        "Experian",
        dict(base, past_due_amount=0.0, two_year_payment_history="OK,30,OK,60"),
    )
    return session_id


def _decision(session_id, acc_id):
    case = cs_api.get_account_case(session_id, acc_id)
    return case.artifacts["stageA_detection"].model_dump()


def test_adopt_ai(monkeypatch, session_case):
    monkeypatch.setattr(config, "ENABLE_CASESTORE_STAGEA", True)
    monkeypatch.setattr(config, "ENABLE_AI_ADJUDICATOR", True)

    resp = AIAdjudicateResponse(
        primary_issue="collection",
        tier="Tier1",
        confidence=0.82,
        problem_reasons=["ai_reason"],
        fields_used=["balance_owed"],
    )
    monkeypatch.setattr(pd, "call_adjudicator", lambda session, req: resp)

    pd.run_stage_a(session_case, [])
    dec1 = _decision(session_case, "acc1")
    assert dec1["decision_source"] == "ai"
    assert dec1["primary_issue"] == "collection"
    assert dec1["tier"] == "Tier1"
    assert dec1["problem_reasons"] == ["ai_reason"]


def test_low_confidence_fallback(monkeypatch, session_case):
    monkeypatch.setattr(config, "ENABLE_CASESTORE_STAGEA", True)
    monkeypatch.setattr(config, "ENABLE_AI_ADJUDICATOR", True)

    resp = AIAdjudicateResponse(
        primary_issue="collection",
        tier="Tier1",
        confidence=0.4,
        problem_reasons=["ai_reason"],
        fields_used=["balance_owed"],
    )
    monkeypatch.setattr(pd, "call_adjudicator", lambda session, req: resp)

    pd.run_stage_a(session_case, [])
    dec2 = _decision(session_case, "acc2")
    assert dec2["decision_source"] == "rules"
    assert dec2["problem_reasons"] == ["past_due_amount: 125.00"]


def test_timeout_fallback(monkeypatch, session_case):
    monkeypatch.setattr(config, "ENABLE_CASESTORE_STAGEA", True)
    monkeypatch.setattr(config, "ENABLE_AI_ADJUDICATOR", True)

    monkeypatch.setattr(pd, "call_adjudicator", lambda session, req: None)

    pd.run_stage_a(session_case, [])
    dec3 = _decision(session_case, "acc3")
    assert dec3["decision_source"] == "rules"
    assert dec3["problem_reasons"] == ["late: 1×30,1×60"]


def test_idempotent(monkeypatch, session_case):
    monkeypatch.setattr(config, "ENABLE_CASESTORE_STAGEA", True)
    monkeypatch.setattr(config, "ENABLE_AI_ADJUDICATOR", True)
    monkeypatch.setattr(pd, "call_adjudicator", lambda session, req: None)

    pd.run_stage_a(session_case, [])
    first = _decision(session_case, "acc2")
    pd.run_stage_a(session_case, [])
    second = _decision(session_case, "acc2")
    first.pop("timestamp", None)
    second.pop("timestamp", None)
    assert first == second
