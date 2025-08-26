from importlib import reload

from flask import Flask

import backend.api.internal_ai as ai
import backend.config as base_config


def _client(monkeypatch, **env):
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    reload(base_config)
    reload(ai)
    app = Flask(__name__)
    app.register_blueprint(ai.internal_ai_bp)
    return app.test_client()


def test_endpoint_disabled(monkeypatch):
    client = _client(monkeypatch, ENABLE_AI_ADJUDICATOR="0")
    resp = client.post(
        "/internal/ai-adjudicate",
        json={"session_id": "s", "hierarchy_version": "v1", "account": {}},
    )
    data = resp.get_json()
    assert data["primary_issue"] == "unknown"
    assert data["error"] == "Disabled"


def test_endpoint_success(monkeypatch):
    client = _client(monkeypatch, ENABLE_AI_ADJUDICATOR="1")
    payload = {
        "session_id": "s",
        "hierarchy_version": "v1",
        "account": {"account_status": "Collection"},
    }
    resp = client.post("/internal/ai-adjudicate", json=payload)
    data = resp.get_json()
    assert data["primary_issue"] == "collection"
    assert data["confidence"] > 0


def test_endpoint_timeout(monkeypatch):
    client = _client(monkeypatch, ENABLE_AI_ADJUDICATOR="1")

    def boom(*a, **k):
        raise TimeoutError("boom")

    monkeypatch.setattr(ai, "_basic_model", boom)
    resp = client.post(
        "/internal/ai-adjudicate",
        json={"session_id": "s", "hierarchy_version": "v1", "account": {}},
    )
    data = resp.get_json()
    assert data["primary_issue"] == "unknown"
    assert data["error"] == "Timeout"
