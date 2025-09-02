import importlib

import pytest

import backend.config as config
from backend.core.case_store import api as cs_api
from backend.core.logic.report_analysis import problem_detection as pd


def _setup_env(monkeypatch, tmp_path, one_case=True, normalized=False):
    monkeypatch.setattr(config, "CASESTORE_DIR", str(tmp_path))
    monkeypatch.setattr(config, "ENABLE_CASESTORE_STAGEA", True)
    monkeypatch.setattr(config, "ENABLE_AI_ADJUDICATOR", False)
    monkeypatch.setattr(config, "ENABLE_CANDIDATE_TOKEN_LOGGER", False)
    monkeypatch.setenv("SAFE_MERGE_ENABLED", "1")
    monkeypatch.setenv("ONE_CASE_PER_ACCOUNT_ENABLED", "1" if one_case else "0")
    monkeypatch.setenv("NORMALIZED_OVERLAY_ENABLED", "1" if normalized else "0")
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    import backend.core.config.flags as flags
    importlib.reload(flags)
    importlib.reload(cs_api)
    importlib.reload(pd)
    import backend.core.materialize.casestore_view as cs_view
    importlib.reload(cs_view)
    import backend.api.app as app_module
    importlib.reload(app_module)
    return app_module


def _create_account(session_id, bureaus, include_normalized=False):
    case = cs_api.create_session_case(session_id)
    cs_api.save_session_case(case)
    fields = {"by_bureau": {}}
    for idx, b in enumerate(bureaus, 1):
        fields["by_bureau"][b] = {
            "balance_owed": idx,
            "payment_status": "late",
            "credit_limit": 1000,
        }
    if include_normalized:
        fields["normalized"] = {"foo": "bar"}
    cs_api.upsert_account_fields(session_id, "acc1", "Experian", fields)


def _create_legacy_account(session_id):
    case = cs_api.create_session_case(session_id)
    cs_api.save_session_case(case)
    fields = {"balance_owed": 50, "payment_status": "late"}
    cs_api.upsert_account_fields(session_id, "acc1", "Experian", fields)


def test_returns_by_bureau_and_artifacts_when_flag_on(tmp_path, monkeypatch):
    app_module = _setup_env(monkeypatch, tmp_path, one_case=True)
    session_id = "sess1"
    _create_account(session_id, ["EX", "EQ", "TU"])
    pd.run_stage_a(session_id)
    app = app_module.create_app()
    client = app.test_client()
    resp = client.get(f"/api/account/{session_id}/acc1")
    assert resp.status_code == 200
    data = resp.get_json()
    assert set(data["fields"]["by_bureau"].keys()) == {"EX", "EQ", "TU"}
    for b in ["EX", "EQ", "TU"]:
        assert f"stageA_detection.{b}" in data["artifacts"]
    assert data["meta"]["flags"]["one_case_per_account_enabled"] is True


def test_handles_missing_bureau_gracefully(tmp_path, monkeypatch):
    app_module = _setup_env(monkeypatch, tmp_path, one_case=True)
    session_id = "sess2"
    _create_account(session_id, ["EX", "TU"])
    pd.run_stage_a(session_id)
    app = app_module.create_app()
    client = app.test_client()
    resp = client.get(f"/api/account/{session_id}/acc1")
    assert resp.status_code == 200
    data = resp.get_json()
    assert "EQ" not in data["fields"]["by_bureau"]
    assert "stageA_detection.EQ" not in data["artifacts"]
    assert data["meta"]["present_bureaus"] == ["EX", "TU"]


def test_includes_normalized_if_present(tmp_path, monkeypatch):
    app_module = _setup_env(monkeypatch, tmp_path, one_case=True, normalized=True)
    session_id = "sess3"
    _create_account(session_id, ["EX"], include_normalized=True)
    pd.run_stage_a(session_id)
    app = app_module.create_app()
    client = app.test_client()
    resp = client.get(f"/api/account/{session_id}/acc1")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["fields"]["normalized"] == {"foo": "bar"}
    assert data["meta"]["flags"]["normalized_overlay_enabled"] is True


def test_legacy_case_without_by_bureau_falls_back_cleanly(tmp_path, monkeypatch):
    app_module = _setup_env(monkeypatch, tmp_path, one_case=False)
    session_id = "sess4"
    _create_legacy_account(session_id)
    pd.run_stage_a(session_id)
    app = app_module.create_app()
    client = app.test_client()
    resp = client.get(f"/api/account/{session_id}/acc1")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["fields"]["balance_owed"] == 50
    assert "stageA_detection" in data["artifacts"]
    assert not any(k.startswith("stageA_detection.") for k in data["artifacts"])


def test_404_when_account_missing(tmp_path, monkeypatch):
    app_module = _setup_env(monkeypatch, tmp_path, one_case=True)
    session_id = "sess5"
    case = cs_api.create_session_case(session_id)
    cs_api.save_session_case(case)
    app = app_module.create_app()
    client = app.test_client()
    resp = client.get(f"/api/account/{session_id}/nope")
    assert resp.status_code == 404
    assert resp.get_json()["error"] == "account_not_found"
