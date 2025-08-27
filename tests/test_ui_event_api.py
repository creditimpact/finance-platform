import json
import re

import pytest

from backend.api.app import create_app
from backend.core.telemetry import ui_ingest

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b")
SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")


def _assert_no_pii(events):
    payload = json.dumps([fields for _, fields in events])
    for pat in (EMAIL_RE, PHONE_RE, SSN_RE):
        assert not pat.search(payload)


@pytest.fixture
def app():
    return create_app()


def test_ui_event_happy_path(app, monkeypatch):
    client = app.test_client()
    events = []
    monkeypatch.setattr(ui_ingest, "telemetry_emit", lambda e, p: events.append((e, p)))
    payload = {
        "type": "ui_review_expand",
        "ts": "2024-01-01T00:00:00Z",
        "session_id": "sess1",
        "payload": {
            "account_id": "acc1",
            "bureau": "Experian",
            "decision_source": "ai",
            "tier": "Tier1",
        },
    }
    resp = client.post("/api/ui-event", json=payload)
    assert resp.status_code == 204
    assert events == [
        (
            "ui_review_expand",
            {
                "session_id": "sess1",
                "account_id": "acc1",
                "bureau": "Experian",
                "decision_source": "ai",
                "tier": "Tier1",
                "ts": "2024-01-01T00:00:00Z",
            },
        )
    ]
    _assert_no_pii(events)


def test_ui_event_invalid_schema(app):
    client = app.test_client()
    bad_payload = {
        "type": "ui_review_expand",
        "ts": "2024-01-01T00:00:00Z",
        "session_id": "sess1",
        "payload": {
            "account_id": "acc1",
            "bureau": "Experian",
            "decision_source": "maybe",
        },
    }
    resp = client.post("/api/ui-event", json=bad_payload)
    assert resp.status_code == 400
    assert resp.get_json()["error"] == "SchemaValidationError"


def test_ui_event_body_size_cap(app):
    client = app.test_client()
    big_session = "a" * 5000
    payload = {
        "type": "ui_review_expand",
        "ts": "2024-01-01T00:00:00Z",
        "session_id": big_session,
        "payload": {"account_id": "acc1", "bureau": "EX"},
    }
    resp = client.post("/api/ui-event", json=payload)
    assert resp.status_code == 413


def test_ui_event_unknown_type(app):
    client = app.test_client()
    payload = {
        "type": "ui_unknown",
        "ts": "2024-01-01T00:00:00Z",
        "session_id": "sess1",
        "payload": {"account_id": "acc1", "bureau": "EX"},
    }
    resp = client.post("/api/ui-event", json=payload)
    assert resp.status_code == 400
    assert resp.get_json()["error"] == "SchemaValidationError"
