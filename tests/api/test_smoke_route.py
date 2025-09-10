import importlib

import pytest


@pytest.fixture
def app(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.com/v1")
    import backend.api.routes_smoke as routes_smoke
    calls = {}

    def dummy_run_full_pipeline(sid):
        calls["sid"] = sid
        return {"queued": True}

    monkeypatch.setattr(routes_smoke, "run_full_pipeline", dummy_run_full_pipeline)
    import backend.api.app as app_module
    importlib.reload(app_module)
    return app_module.create_app(), calls


def test_smoke_run_success(app):
    app_instance, calls = app
    client = app_instance.test_client()
    resp = client.post("/smoke/run", json={"sid": "123"})
    assert resp.status_code == 200
    assert resp.get_json() == {"sid": "123", "queued": True}
    assert calls["sid"] == "123"


def test_smoke_run_missing_sid(app):
    app_instance, _ = app
    client = app_instance.test_client()
    resp = client.post("/smoke/run", json={})
    assert resp.status_code == 400
    assert resp.get_json()["error"] == "sid required"
