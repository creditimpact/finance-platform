import logging
from pathlib import Path

import pytest

import backend.config as config
from backend.core.case_store import api as cs_api
from backend.core import orchestrators as orch
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
    cs_api.upsert_account_fields(session_id, "acc1", "Experian", dict(base, past_due_amount=0.0))
    cs_api.upsert_account_fields(
        session_id, "acc2", "Experian", dict(base, past_due_amount=125.0)
    )
    cs_api.upsert_account_fields(
        session_id,
        "acc3",
        "Experian",
        dict(base, past_due_amount=0.0, two_year_payment_history="OK,30,OK,60,OK,OK,30"),
    )
    return session_id


def _decision(session_id, acc_id):
    case = cs_api.get_account_case(session_id, acc_id)
    return case.artifacts["stageA_detection"].model_dump()


def test_detection_only_neutral(monkeypatch, session_case):
    monkeypatch.setattr(config, "ENABLE_CASESTORE_STAGEA", True)
    monkeypatch.setattr(config, "PROBLEM_DETECTION_ONLY", True)
    pd.run_stage_a(session_case, [])
    problems = orch.collect_stageA_problem_accounts(session_case, [])
    ids = {p["account_id"] for p in problems}
    assert ids == {"acc2", "acc3"}
    dec2 = _decision(session_case, "acc2")
    assert dec2["primary_issue"] == "unknown"
    assert dec2["tier"] == "none"
    assert dec2["decision_source"] == "rules"
    assert dec2["confidence"] == 0.0
    assert dec2["problem_reasons"] == ["past_due_amount: 125.00"]
    dec3 = _decision(session_case, "acc3")
    assert dec3["problem_reasons"] == ["late: 2×30,1×60"]


def test_idempotent(monkeypatch, session_case):
    monkeypatch.setattr(config, "ENABLE_CASESTORE_STAGEA", True)
    pd.run_stage_a(session_case, [])
    first = _decision(session_case, "acc2")
    pd.run_stage_a(session_case, [])
    second = _decision(session_case, "acc2")
    first.pop("timestamp", None)
    second.pop("timestamp", None)
    assert first == second


def test_extract_late_counts_tolerates_noise():
    history = ["OK", "30D", "120+", "?", "OK"]
    counts = pd._extract_late_counts(history)
    assert counts == {"30": 1, "120": 1}


def test_no_keyword_tables():
    text = Path(pd.__file__).read_text().lower()
    for bad in ["collection", "charge off", "charge_off", "repossession", "keyword"]:
        assert bad not in text
