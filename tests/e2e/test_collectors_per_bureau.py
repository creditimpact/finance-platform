import importlib

import pytest

import backend.config as config
from backend.core.case_store import api as cs_api
from backend.core.logic.report_analysis import problem_detection as pd

ALLOWED_KEYS = {
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


@pytest.fixture(autouse=True)
def reset_flags(monkeypatch):
    yield
    monkeypatch.setenv("ONE_CASE_PER_ACCOUNT_ENABLED", "0")
    import backend.core.config.flags as flags
    import backend.core.orchestrators as orch
    importlib.reload(flags)
    importlib.reload(orch)


def _setup_env(monkeypatch, tmp_path, flag_on: bool) -> None:
    monkeypatch.setattr(config, "CASESTORE_DIR", str(tmp_path))
    monkeypatch.setattr(config, "ENABLE_CASESTORE_STAGEA", True)
    monkeypatch.setattr(config, "ENABLE_AI_ADJUDICATOR", False)
    monkeypatch.setattr(config, "ENABLE_CANDIDATE_TOKEN_LOGGER", False)
    monkeypatch.setenv("SAFE_MERGE_ENABLED", "1")
    monkeypatch.setenv("ONE_CASE_PER_ACCOUNT_ENABLED", "1" if flag_on else "0")
    import backend.core.config.flags as flags
    import backend.core.orchestrators as orch
    importlib.reload(flags)
    importlib.reload(cs_api)
    importlib.reload(pd)
    importlib.reload(orch)


def _create_session(monkeypatch, tmp_path, flag_on: bool) -> str:
    _setup_env(monkeypatch, tmp_path, flag_on)
    session_id = "sess1"
    case = cs_api.create_session_case(session_id)
    cs_api.save_session_case(case)
    fields = {
        "by_bureau": {
            "EX": {"balance_owed": 1, "credit_limit": 1000, "past_due_amount": 10},
            "EQ": {"balance_owed": 2, "credit_limit": 1000, "past_due_amount": 20},
            "TU": {"balance_owed": 3, "credit_limit": 1000, "past_due_amount": 30},
        }
    }
    cs_api.upsert_account_fields(session_id, "acc1", "Experian", fields)
    return session_id


def _assert_shape(records):
    for rec in records:
        assert set(rec.keys()) <= ALLOWED_KEYS


def test_collect_stageA_problem_accounts_reads_namespaced_when_flag_on(tmp_path, monkeypatch):
    session_id = _create_session(monkeypatch, tmp_path, True)
    pd.run_stage_a(session_id)
    import backend.core.orchestrators as orch

    rows = orch.collect_stageA_problem_accounts(session_id)
    assert len(rows) == 3
    bureaus = {r["bureau"] for r in rows}
    assert bureaus == {"EX", "EQ", "TU"}
    for r in rows:
        assert r["account_id"] == "acc1"
    _assert_shape(rows)


def test_collect_stageA_problem_accounts_falls_back_to_legacy(tmp_path, monkeypatch):
    _setup_env(monkeypatch, tmp_path, False)
    session_id = "sess1"
    case = cs_api.create_session_case(session_id)
    cs_api.save_session_case(case)
    root_fields = {"balance_owed": 1, "credit_limit": 1000, "past_due_amount": 10}
    cs_api.upsert_account_fields(session_id, "acc1", "Experian", root_fields)
    pd.run_stage_a(session_id)
    import backend.core.orchestrators as orch

    rows = orch.collect_stageA_problem_accounts(session_id)
    assert len(rows) == 1
    assert rows[0]["bureau"] == "Experian"
    _assert_shape(rows)


def test_collect_stageA_logical_accounts_winner_is_stable(tmp_path, monkeypatch):
    session_id = _create_session(monkeypatch, tmp_path, True)
    monkeypatch.setattr(config, "ENABLE_AI_ADJUDICATOR", True)
    monkeypatch.setattr(config, "ENABLE_CROSS_BUREAU_RESOLUTION", True)

    def fake_eval_with_optional_ai(session_id, account_id, fields, doc_fp, acct_fp):
        bal = fields.get("balance_owed")
        mapping = {
            1: ("moderate_delinquency", 0.4),
            2: ("severe_delinquency", 0.6),
            3: ("severe_delinquency", 0.9),
        }
        issue, conf = mapping.get(bal, ("unknown", 0.0))
        decision = {
            "primary_issue": issue,
            "tier": "none",
            "confidence": conf,
            "decision_source": "ai",
            "problem_reasons": [],
        }
        return decision, True, 0.0, None, conf

    monkeypatch.setattr(pd, "evaluate_with_optional_ai", fake_eval_with_optional_ai)
    pd.run_stage_a(session_id)
    import backend.core.orchestrators as orch

    records = orch.collect_stageA_logical_accounts(session_id)
    assert len(records) == 1
    winner = records[0]
    assert winner["bureau"] == "TU"
    assert winner["tier"] == "Tier2"
    assert winner["confidence"] == pytest.approx(0.9)
    _assert_shape(records)

    case = cs_api.get_account_case(session_id, "acc1")
    legacy = case.artifacts["stageA_detection"].model_dump()
    assert winner["tier"] == legacy["tier"]
    assert winner["confidence"] == legacy["confidence"]
    assert winner["decision_source"] == legacy["decision_source"]
    assert winner["primary_issue"] == legacy["primary_issue"]


def test_output_shape_unchanged_for_api(tmp_path, monkeypatch):
    session_id = _create_session(monkeypatch, tmp_path, True)
    monkeypatch.setattr(config, "ENABLE_CROSS_BUREAU_RESOLUTION", True)
    pd.run_stage_a(session_id)
    import backend.core.orchestrators as orch

    rows = orch.collect_stageA_problem_accounts(session_id)
    winners = orch.collect_stageA_logical_accounts(session_id)
    _assert_shape(rows)
    _assert_shape(winners)
