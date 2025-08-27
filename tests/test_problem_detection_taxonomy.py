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
        session_id, "acc1", "Experian", dict(base, past_due_amount=125.0)
    )
    return session_id


def _decision(session_id: str):
    case = cs_api.get_account_case(session_id, "acc1")
    return case.artifacts["stageA_detection"].model_dump()


def test_ai_mismatch_fix(monkeypatch, session_case):
    monkeypatch.setattr(config, "ENABLE_CASESTORE_STAGEA", True)
    monkeypatch.setattr(config, "ENABLE_AI_ADJUDICATOR", True)

    resp = AIAdjudicateResponse(
        primary_issue="collection",
        tier="Tier3",
        confidence=0.9,
        problem_reasons=["ai_reason"],
        fields_used=["balance_owed"],
    )
    monkeypatch.setattr(pd, "call_adjudicator", lambda session, req: resp)

    pd.run_stage_a(session_case, [])
    dec = _decision(session_case)
    assert dec["decision_source"] == "ai"
    assert dec["primary_issue"] == "collection"
    assert dec["tier"] == "Tier1"


def test_rules_neutral_preserves_none(monkeypatch, session_case):
    monkeypatch.setattr(config, "ENABLE_CASESTORE_STAGEA", True)
    monkeypatch.setattr(config, "ENABLE_AI_ADJUDICATOR", False)

    pd.run_stage_a(session_case, [])
    dec = _decision(session_case)
    assert dec["decision_source"] == "rules"
    assert dec["primary_issue"] == "unknown"
    assert dec["tier"] == "none"


def test_ai_high_utilization_tier4(monkeypatch, session_case):
    monkeypatch.setattr(config, "ENABLE_CASESTORE_STAGEA", True)
    monkeypatch.setattr(config, "ENABLE_AI_ADJUDICATOR", True)

    resp = AIAdjudicateResponse(
        primary_issue="high_utilization",
        tier="Tier2",
        confidence=0.9,
        problem_reasons=["ai_reason"],
        fields_used=["balance_owed"],
    )
    monkeypatch.setattr(pd, "call_adjudicator", lambda session, req: resp)

    pd.run_stage_a(session_case, [])
    dec = _decision(session_case)
    assert dec["decision_source"] == "ai"
    assert dec["primary_issue"] == "high_utilization"
    assert dec["tier"] == "Tier4"
