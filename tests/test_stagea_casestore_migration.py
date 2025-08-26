import os
import time
import logging

import pytest

import backend.config as config
from backend.core.case_store import api as cs_api
from backend.core.logic.report_analysis import problem_detection as pd
from backend.core import orchestrators as orch


@pytest.fixture
def session_case(tmp_path, monkeypatch):
    """Create a synthetic Case Store session with two accounts."""
    session_id = "sess1"
    monkeypatch.setattr(config, "CASESTORE_DIR", str(tmp_path))
    case = cs_api.create_session_case(session_id)
    cs_api.save_session_case(case)
    cs_api.upsert_account_fields(
        session_id,
        "acc1",
        "Experian",
        {"past_due_amount": 50.0, "balance_owed": 100.0},
    )
    cs_api.upsert_account_fields(
        session_id,
        "acc2",
        "Equifax",
        {"past_due_amount": 0.0, "balance_owed": 100.0},
    )
    legacy_accounts = [
        {"account_id": "acc1", "past_due_amount": 50.0},
        {"account_id": "acc2", "past_due_amount": 0.0},
    ]
    return session_id, legacy_accounts


def test_legacy_path(session_case, monkeypatch):
    session_id, legacy_accounts = session_case
    monkeypatch.setattr(config, "ENABLE_CASESTORE_STAGEA", False)
    pd.run_stage_a(session_id, legacy_accounts)
    problems = orch.collect_stageA_problem_accounts(session_id, legacy_accounts)
    assert [p["account_id"] for p in problems] == ["acc1"]


def test_casestore_path(session_case, monkeypatch):
    session_id, legacy_accounts = session_case
    monkeypatch.setattr(config, "ENABLE_CASESTORE_STAGEA", True)
    pd.run_stage_a(session_id, legacy_accounts)
    # artifacts written
    for aid in ["acc1", "acc2"]:
        case = cs_api.get_account_case(session_id, aid)
        assert "stageA_detection" in case.artifacts
    problems = orch.collect_stageA_problem_accounts(session_id, [])
    assert [p["account_id"] for p in problems] == ["acc1"]


def test_parity_logging(session_case, monkeypatch, caplog):
    session_id, legacy_accounts = session_case
    monkeypatch.setattr(config, "ENABLE_CASESTORE_STAGEA", True)
    monkeypatch.setattr(config, "CASESTORE_STAGEA_LOG_PARITY", True)
    caplog.set_level(logging.INFO)
    pd.run_stage_a(session_id, legacy_accounts)
    parity_logs = [r for r in caplog.records if "stageA_parity" in r.message]
    assert len(parity_logs) == 2


def test_missing_account_resilience(session_case, monkeypatch, caplog):
    session_id, legacy_accounts = session_case
    monkeypatch.setattr(config, "ENABLE_CASESTORE_STAGEA", True)
    caplog.set_level(logging.WARNING)

    orig = pd.get_account_fields

    def broken_get_fields(session_id, account_id, fields):
        if account_id == "acc2":
            raise cs_api.CaseStoreError(code=cs_api.NOT_FOUND, message="missing")
        return orig(session_id, account_id, fields)

    monkeypatch.setattr(pd, "get_account_fields", broken_get_fields)
    pd.run_stage_a(session_id, legacy_accounts)
    case1 = cs_api.get_account_case(session_id, "acc1")
    assert "stageA_detection" in case1.artifacts
    case2 = cs_api.get_account_case(session_id, "acc2")
    assert "stageA_detection" not in case2.artifacts
    assert any("stageA_missing_account" in r.message for r in caplog.records)


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
