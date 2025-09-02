import importlib

import backend.config as config
from backend.core.case_store import api as cs_api
from backend.core.logic.report_analysis import problem_detection as pd


def _setup_env(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "CASESTORE_DIR", str(tmp_path))
    monkeypatch.setattr(config, "ENABLE_CASESTORE_STAGEA", True)
    monkeypatch.setenv("SAFE_MERGE_ENABLED", "1")
    monkeypatch.setenv("ONE_CASE_PER_ACCOUNT_ENABLED", "1")
    monkeypatch.setenv("CASE_FIRST_BUILD_REQUIRED", "1")
    monkeypatch.setenv("DISABLE_PARSER_UI_SUMMARY", "1")
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    import backend.core.config.flags as flags
    importlib.reload(flags)
    importlib.reload(cs_api)
    importlib.reload(pd)
    import backend.api.app as app_module
    importlib.reload(app_module)
    return app_module


def _create_account(session_id):
    case = cs_api.create_session_case(session_id)
    cs_api.save_session_case(case)
    fields = {"by_bureau": {"EX": {"balance_owed": 1, "payment_status": "late"}}}
    cs_api.upsert_account_fields(session_id, "acc1", "Experian", fields)


def test_no_cases_returns_empty(tmp_path, monkeypatch):
    app_module = _setup_env(monkeypatch, tmp_path)
    session_id = "sess1"
    case = cs_api.create_session_case(session_id)
    cs_api.save_session_case(case)
    app = app_module.create_app()
    client = app.test_client()
    resp = client.get(f"/api/accounts/{session_id}")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["accounts"] == []


def test_collects_from_case_store(tmp_path, monkeypatch):
    app_module = _setup_env(monkeypatch, tmp_path)
    session_id = "sess2"
    _create_account(session_id)
    pd.run_stage_a(session_id)
    app = app_module.create_app()
    client = app.test_client()
    resp = client.get(f"/api/accounts/{session_id}")
    assert resp.status_code == 200
    accounts = resp.get_json()["accounts"]
    assert accounts
    assert all(a.get("source_stage") != "parser_aggregated" for a in accounts)
