import logging
import time

import pytest

import backend.config as config
from backend.core import orchestrators as orch
from backend.core.case_store import api as cs_api
from backend.core.logic.report_analysis import problem_detection as pd


@pytest.fixture
def session_case(tmp_path, monkeypatch):
    """Create a synthetic Case Store session with three accounts."""
    session_id = "sess1"
    monkeypatch.setattr(config, "CASESTORE_DIR", str(tmp_path))
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
    # Clean account
    cs_api.upsert_account_fields(session_id, "acc1", "Experian", dict(base))
    # Numeric past-due evidence
    cs_api.upsert_account_fields(
        session_id, "acc2", "Experian", dict(base, past_due_amount=50.0)
    )
    # Late-history evidence
    cs_api.upsert_account_fields(
        session_id,
        "acc3",
        "Experian",
        dict(base, two_year_payment_history="OK,30,OK,60,OK"),
    )
    legacy_accounts = [
        {"account_id": "acc1"},
        {"account_id": "acc2", "past_due_amount": 50.0},
        {
            "account_id": "acc3",
            "two_year_payment_history": "OK,30,OK,60,OK",
            "past_due_amount": 0.0,
        },
    ]
    return session_id, legacy_accounts


def test_legacy_path(session_case, monkeypatch):
    session_id, legacy_accounts = session_case
    monkeypatch.setattr(config, "ENABLE_CASESTORE_STAGEA", False)
    pd.run_stage_a(session_id, legacy_accounts)
    problems = orch.collect_stageA_problem_accounts(session_id, legacy_accounts)
    assert {p["account_id"] for p in problems} == {"acc2", "acc3"}


def test_casestore_path(session_case, monkeypatch):
    session_id, legacy_accounts = session_case
    monkeypatch.setattr(config, "ENABLE_CASESTORE_STAGEA", True)
    pd.run_stage_a(session_id, legacy_accounts)
    # artifacts written
    for aid in ["acc1", "acc2", "acc3"]:
        case = cs_api.get_account_case(session_id, aid)
        assert "stageA_detection" in case.artifacts
    problems = orch.collect_stageA_problem_accounts(session_id, [])
    ids = {p["account_id"] for p in problems}
    assert ids == {"acc2", "acc3"}
    for acc in problems:
        assert acc["primary_issue"] == "unknown"
        assert acc["tier"] == "none"
        assert acc["decision_source"] == "rules"
        assert acc["confidence"] == 0.0
    reasons = {p["account_id"]: p["problem_reasons"] for p in problems}
    assert reasons["acc2"] == ["past_due_amount: 50.00"]
    assert reasons["acc3"] == ["late: 1×30,1×60"]


def test_parity_logging(session_case, monkeypatch, caplog):
    session_id, legacy_accounts = session_case
    monkeypatch.setattr(config, "ENABLE_CASESTORE_STAGEA", True)
    monkeypatch.setattr(config, "CASESTORE_STAGEA_LOG_PARITY", True)
    caplog.set_level(logging.INFO)
    pd.run_stage_a(session_id, legacy_accounts)
    parity_logs = [r for r in caplog.records if "stageA_parity" in r.message]
    assert len(parity_logs) == 3


def test_idempotent(session_case, monkeypatch):
    session_id, legacy_accounts = session_case
    monkeypatch.setattr(config, "ENABLE_CASESTORE_STAGEA", True)
    pd.run_stage_a(session_id, legacy_accounts)
    first = (
        cs_api.get_account_case(session_id, "acc2")
        .artifacts["stageA_detection"]
        .model_dump()
    )
    pd.run_stage_a(session_id, legacy_accounts)
    second = (
        cs_api.get_account_case(session_id, "acc2")
        .artifacts["stageA_detection"]
        .model_dump()
    )
    first.pop("timestamp", None)
    second.pop("timestamp", None)
    assert first == second


def test_missing_account_resilience(session_case, monkeypatch, caplog):
    session_id, legacy_accounts = session_case
    monkeypatch.setattr(config, "ENABLE_CASESTORE_STAGEA", True)
    caplog.set_level(logging.WARNING)

    orig = pd.append_artifact

    def broken_append(session_id, acc_id, namespace, payload, **kw):
        if acc_id == "acc3":
            raise cs_api.CaseStoreError(code=cs_api.NOT_FOUND, message="boom")
        return orig(session_id, acc_id, namespace, payload, **kw)

    monkeypatch.setattr(pd, "append_artifact", broken_append)
    pd.run_stage_a(session_id, legacy_accounts)
    case1 = cs_api.get_account_case(session_id, "acc1")
    assert "stageA_detection" in case1.artifacts
    case2 = cs_api.get_account_case(session_id, "acc2")
    assert "stageA_detection" in case2.artifacts
    case3 = cs_api.get_account_case(session_id, "acc3")
    assert "stageA_detection" not in case3.artifacts
    assert any("stageA_append_failed" in r.message for r in caplog.records)


def test_performance_sanity(tmp_path, monkeypatch):
    session_id = "perf"
    monkeypatch.setattr(config, "CASESTORE_DIR", str(tmp_path))
    case = cs_api.create_session_case(session_id)
    cs_api.save_session_case(case)
    for i in range(50):
        cs_api.upsert_account_fields(
            session_id,
            f"a{i}",
            "Experian",
            {"past_due_amount": float(i % 2), "balance_owed": 100.0},
        )
    monkeypatch.setattr(config, "ENABLE_CASESTORE_STAGEA", True)
    start = time.time()
    pd.run_stage_a(session_id, [])
    dur = time.time() - start
    assert dur < 2.0
