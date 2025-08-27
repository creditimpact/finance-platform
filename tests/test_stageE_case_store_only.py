import io
import json
import uuid

import pytest

import backend.config as config
from backend.api.app import create_app
from backend.api import app as app_module
from backend.core.case_store import api as cs_api
from backend.core.case_store.errors import CaseStoreError, IO_ERROR


class DummyResult:
    def get(self, timeout=None):
        return {"problem_accounts": [{"account_id": "legacy"}]}


class DummyTask:
    def delay(self, *a, **k):
        return DummyResult()


def _setup_case(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CASESTORE_DIR", str(tmp_path))
    monkeypatch.setattr(app_module, "set_session", lambda *a, **k: None)
    monkeypatch.setattr(app_module, "extract_problematic_accounts", DummyTask())
    monkeypatch.setattr(config, "API_INCLUDE_DECISION_META", False)

    session_id = "sess1"

    class DummyUUID:
        hex = "filehex"

        def __str__(self):
            return session_id

    monkeypatch.setattr(uuid, "uuid4", lambda: DummyUUID())

    case = cs_api.create_session_case(session_id)
    cs_api.save_session_case(case)
    cs_api.upsert_account_fields(session_id, "acc1", "Experian", {})
    cs_api.append_artifact(
        session_id,
        "acc1",
        "stageA_detection",
        {
            "primary_issue": "collection",
            "tier": "Tier1",
            "confidence": 0.9,
            "problem_reasons": ["ai"],
            "decision_source": "ai",
        },
    )
    return session_id


def test_api_uses_case_store_only(tmp_path, monkeypatch):
    _setup_case(tmp_path, monkeypatch)
    app = create_app()
    client = app.test_client()
    data = {"email": "a@example.com", "file": (io.BytesIO(b"%PDF-1.4"), "r.pdf")}
    resp = client.post("/api/start-process", data=data, content_type="multipart/form-data")
    assert resp.status_code == 200
    payload = json.loads(resp.data)
    accounts = payload["accounts"]["problem_accounts"]
    assert accounts[0]["account_id"] == "acc1"
    assert accounts[0]["account_id"] != "legacy"


def test_api_errors_when_casestore_unavailable(tmp_path, monkeypatch):
    _setup_case(tmp_path, monkeypatch)
    app = create_app()
    client = app.test_client()
    monkeypatch.setattr(
        app_module.cs_api,
        "load_session_case",
        lambda sid: (_ for _ in ()).throw(CaseStoreError(code=IO_ERROR, message="boom")),
    )
    data = {"email": "a@example.com", "file": (io.BytesIO(b"%PDF-1.4"), "r.pdf")}
    resp = client.post("/api/start-process", data=data, content_type="multipart/form-data")
    assert resp.status_code >= 500
    payload = json.loads(resp.data)
    assert payload["status"] == "error"
