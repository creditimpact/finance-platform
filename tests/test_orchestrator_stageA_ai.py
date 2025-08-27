from __future__ import annotations

import pytest

import backend.config as config
from backend.core import orchestrators as orch
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


def test_orchestrator_filters_ai_tiers(monkeypatch, session_case):
    monkeypatch.setattr(config, "ENABLE_CASESTORE_STAGEA", True)
    monkeypatch.setattr(config, "ENABLE_AI_ADJUDICATOR", True)

    responses = [
        AIAdjudicateResponse(
            primary_issue="collection",
            tier="Tier1",
            confidence=0.9,
            problem_reasons=["ai_reason"],
            fields_used=["balance_owed"],
        ),
        AIAdjudicateResponse(
            primary_issue="high_utilization",
            tier="Tier2",
            confidence=0.9,
            problem_reasons=["ai_reason"],
            fields_used=["balance_owed"],
        ),
        None,
    ]

    def fake_call(session, req):
        return responses.pop(0)

    monkeypatch.setattr(pd, "call_adjudicator", fake_call)

    pd.run_stage_a(session_case, [])
    problems = orch.collect_stageA_problem_accounts(session_case, [])
    ids = {p["account_id"] for p in problems}
    assert ids == {"acc1", "acc3"}


def test_orchestrator_fallback(monkeypatch, session_case):
    monkeypatch.setattr(config, "ENABLE_CASESTORE_STAGEA", True)
    monkeypatch.setattr(config, "ENABLE_AI_ADJUDICATOR", True)

    monkeypatch.setattr(pd, "call_adjudicator", lambda session, req: None)

    pd.run_stage_a(session_case, [])
    problems = orch.collect_stageA_problem_accounts(session_case, [])
    ids = {p["account_id"] for p in problems}
    assert ids == {"acc2", "acc3"}


def test_orchestrator_low_confidence(monkeypatch, session_case):
    monkeypatch.setattr(config, "ENABLE_CASESTORE_STAGEA", True)
    monkeypatch.setattr(config, "ENABLE_AI_ADJUDICATOR", True)

    responses = [
        AIAdjudicateResponse(
            primary_issue="collection",
            tier="Tier1",
            confidence=0.5,
            problem_reasons=["ai_reason"],
            fields_used=["balance_owed"],
        ),
        AIAdjudicateResponse(
            primary_issue="collection",
            tier="Tier2",
            confidence=0.6,
            problem_reasons=["ai_reason"],
            fields_used=["balance_owed"],
        ),
        None,
    ]

    def fake_call(session, req):
        return responses.pop(0)

    monkeypatch.setattr(pd, "call_adjudicator", fake_call)

    pd.run_stage_a(session_case, [])
    problems = orch.collect_stageA_problem_accounts(session_case, [])
    ids = {p["account_id"] for p in problems}
    assert ids == {"acc2", "acc3"}
