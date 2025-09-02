import importlib
import importlib
from pathlib import Path

import pytest

import backend.config as config
from backend.core.case_store import api as cs_api
from backend.core.case_store.errors import CaseStoreError
from backend.core.logic.report_analysis import problem_detection as pd


def _setup(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(config, "CASESTORE_DIR", tmp_path.as_posix())
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


def _create_account(session_id: str) -> None:
    case = cs_api.create_session_case(session_id)
    cs_api.save_session_case(case)
    fields = {"by_bureau": {"EX": {"balance_owed": 1, "payment_status": "late"}}}
    cs_api.upsert_account_fields(session_id, "acc1", "Experian", fields)


def test_stage_a_requires_cases(tmp_path, monkeypatch):
    app_module = _setup(monkeypatch, tmp_path)
    session_id = "sessA"
    case = cs_api.create_session_case(session_id)
    cs_api.save_session_case(case)
    with pytest.raises(CaseStoreError):
        pd.run_stage_a(session_id)
    app = app_module.create_app()
    client = app.test_client()
    resp = client.get(f"/api/accounts/{session_id}")
    assert resp.status_code == 200
    assert resp.get_json()["accounts"] == []


def test_ui_uses_case_store(tmp_path, monkeypatch):
    app_module = _setup(monkeypatch, tmp_path)
    session_id = "sessB"
    _create_account(session_id)
    pd.run_stage_a(session_id)
    app = app_module.create_app()
    client = app.test_client()
    resp = client.get(f"/api/accounts/{session_id}")
    data = resp.get_json()
    assert len(data["accounts"]) == 1
    account_id = data["accounts"][0]["account_id"]
    resp2 = client.get(f"/api/account/{session_id}/{account_id}")
    assert resp2.status_code == 200
    assert "by_bureau" in resp2.get_json()["fields"]
