from __future__ import annotations

import json

import pytest

from backend.api import ai_endpoints, app as app_module


@pytest.fixture
def client(monkeypatch):
    app = app_module.create_app()
    return app.test_client()


def _request_body():
    return {
        "doc_fingerprint": "doc",
        "account_fingerprint": "acct",
        "fields": {"balance_owed": 1},
    }


def test_ai_endpoint_success(monkeypatch, client):
    def fake_prompt(system, user, *, temperature, timeout):
        return json.dumps(
            {
                "primary_issue": "collection",
                "tier": "Tier1",
                "problem_reasons": ["foo"],
                "confidence": 0.85,
                "fields_used": ["balance_owed"],
            }
        )

    monkeypatch.setattr(ai_endpoints, "run_llm_prompt", fake_prompt)
    resp = client.post("/internal/ai-adjudicate", json=_request_body())
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["primary_issue"] == "collection"
    assert data["tier"] == "Tier1"


def test_ai_endpoint_bad_json(monkeypatch, client):
    def fake_prompt(system, user, *, temperature, timeout):
        return "not json"

    monkeypatch.setattr(ai_endpoints, "run_llm_prompt", fake_prompt)
    resp = client.post("/internal/ai-adjudicate", json=_request_body())
    assert resp.status_code == 502


def test_ai_endpoint_timeout(monkeypatch, client):
    def fake_prompt(system, user, *, temperature, timeout):
        raise TimeoutError

    monkeypatch.setattr(ai_endpoints, "run_llm_prompt", fake_prompt)
    resp = client.post("/internal/ai-adjudicate", json=_request_body())
    assert resp.status_code in {502, 504}


def test_ai_endpoint_schema_validation(monkeypatch, client):
    def fake_prompt(system, user, *, temperature, timeout):
        return json.dumps(
            {
                "primary_issue": "collection",
                "tier": "BadTier",
                "problem_reasons": [],
                "confidence": 0.9,
                "decision_source": "ai",
            }
        )

    monkeypatch.setattr(ai_endpoints, "run_llm_prompt", fake_prompt)
    resp = client.post("/internal/ai-adjudicate", json=_request_body())
    assert resp.status_code == 502
