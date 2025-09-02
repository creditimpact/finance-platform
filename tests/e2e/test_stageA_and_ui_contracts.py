import pytest

import backend.config as config
from backend.core.case_store import api as cs_api
from backend.core.logic.report_analysis import problem_detection as pd
from backend.core.logic.report_analysis.extract_problematic_accounts import (
    extract_problematic_accounts,
)
from backend.core.orchestrators import collect_stageA_problem_accounts

ALLOWED_DECISION_KEYS = {
    "account_id",
    "bureau",
    "primary_issue",
    "tier",
    "problem_reasons",
    "confidence",
    "decision_source",
    "debug",
    "fields_used",
}


class ExplodingAccounts:
    def __len__(self) -> int:  # pragma: no cover - only used for truthiness
        return 0

    def __iter__(self):  # pragma: no cover - should never be invoked
        raise AssertionError("Stage-A attempted to use in-memory accounts")


def _prepare_case(tmp_path, monkeypatch) -> str:
    monkeypatch.setattr(config, "CASESTORE_DIR", str(tmp_path))
    monkeypatch.setattr(config, "ENABLE_CASESTORE_STAGEA", True)
    monkeypatch.setattr(config, "ENABLE_AI_ADJUDICATOR", False)

    session_id = "sess1"
    case = cs_api.create_session_case(session_id)
    cs_api.save_session_case(case)
    base = {"balance_owed": 100.0, "credit_limit": 1000.0}
    cs_api.upsert_account_fields(
        session_id, "acc1", "Experian", dict(base, past_due_amount=0.0)
    )
    cs_api.upsert_account_fields(
        session_id, "acc2", "Experian", dict(base, past_due_amount=50.0)
    )
    return session_id


def _assert_decision_only(records):
    forbidden = {
        "balance_owed",
        "credit_limit",
        "payment_history",
        "by_bureau",
        "normalized",
    }
    for item in records:
        keys = set(item.keys())
        assert (
            keys <= ALLOWED_DECISION_KEYS
        ), f"Unexpected keys: {keys - ALLOWED_DECISION_KEYS}"
        assert not (forbidden & keys), f"Found raw fields: {forbidden & keys}"


@pytest.fixture
def stagea_session(tmp_path, monkeypatch) -> str:
    session_id = _prepare_case(tmp_path, monkeypatch)
    pd.run_stage_a(session_id, [])
    return session_id


def test_stageA_reads_from_case_store_only(tmp_path, monkeypatch):
    session_id = _prepare_case(tmp_path, monkeypatch)

    calls = {"get_fields": 0, "get_case": 0}
    orig_get_fields = pd.get_account_fields
    orig_get_case = cs_api.get_account_case

    def spy_get_fields(*a, **kw):
        calls["get_fields"] += 1
        return orig_get_fields(*a, **kw)

    def spy_get_case(*a, **kw):
        calls["get_case"] += 1
        return orig_get_case(*a, **kw)

    monkeypatch.setattr(pd, "get_account_fields", spy_get_fields)
    monkeypatch.setattr(cs_api, "get_account_case", spy_get_case)

    pd.run_stage_a(session_id, ExplodingAccounts())

    assert (calls["get_fields"] + calls["get_case"]) >= 2


def test_collect_stageA_problem_accounts_decision_only(stagea_session):
    records = collect_stageA_problem_accounts(stagea_session)
    _assert_decision_only(records)


def test_extract_problematic_accounts_decision_only(stagea_session):
    records = extract_problematic_accounts(stagea_session)
    _assert_decision_only(records)
