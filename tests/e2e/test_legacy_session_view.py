import importlib
from pathlib import Path

import backend.config as config
from backend.core.case_store import api as cs_api
import backend.core.compat.legacy_shim as shim

BUREAU_NAMES = {"EX": "Experian", "EQ": "Equifax", "TU": "TransUnion"}


def _setup(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(config, "CASESTORE_DIR", tmp_path.as_posix())
    monkeypatch.setenv("SAFE_MERGE_ENABLED", "1")
    monkeypatch.setenv("ONE_CASE_PER_ACCOUNT_ENABLED", "1")
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    import backend.core.config.flags as flags
    importlib.reload(flags)
    importlib.reload(cs_api)
    importlib.reload(shim)
    import backend.core.materialize.casestore_view as cs_view
    importlib.reload(cs_view)
    import backend.api.app as app_module
    importlib.reload(app_module)
    return app_module


def _create_legacy_session(session_id: str) -> None:
    case = cs_api.create_session_case(session_id)
    cs_api.save_session_case(case)
    base = {
        "account_number": "00001234",
        "creditor_type": "card",
        "date_opened": "2020-01-01",
    }
    for idx, code in enumerate(["EX", "EQ", "TU"], 1):
        fields = dict(base)
        fields["balance_owed"] = idx
        acc_id = f"acc-{code.lower()}"
        cs_api.upsert_account_fields(session_id, acc_id, BUREAU_NAMES[code], fields)
        cs_api.append_artifact(session_id, acc_id, "stageA_detection", {"tier": "none"})


def test_api_view_uses_shim_on_legacy_case(tmp_path, monkeypatch):
    app_module = _setup(monkeypatch, tmp_path)
    session_id = "sess-legacy"
    _create_legacy_session(session_id)
    app = app_module.create_app()
    client = app.test_client()
    resp = client.get(f"/api/account/{session_id}/acc-ex")
    assert resp.status_code == 200
    data = resp.get_json()
    assert set(data["fields"]["by_bureau"].keys()) == {"EX", "EQ", "TU"}
    assert "stageA_detection" in data["artifacts"]
    assert not any(k.startswith("stageA_detection.") for k in data["artifacts"])
    assert data["meta"]["present_bureaus"] == ["EQ", "EX", "TU"]
    case = cs_api.get_account_case(session_id, "acc-ex")
    assert getattr(case.fields, "by_bureau", None) is None
