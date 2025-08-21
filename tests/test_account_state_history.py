from datetime import datetime

import planner
from backend.api import app as api_app, session_manager
from backend.core.models import AccountStatus


def _setup_store(monkeypatch):
    store = {}

    def fake_get_session(sid):
        return store.get(sid)

    def fake_update_session(sid, **kwargs):
        session = store.setdefault(sid, {})
        session.update(kwargs)
        return session

    monkeypatch.setattr(session_manager, "get_session", fake_get_session)
    monkeypatch.setattr(session_manager, "update_session", fake_update_session)
    monkeypatch.setattr(planner, "get_session", fake_get_session)
    monkeypatch.setattr(planner, "update_session", fake_update_session)
    return store


def test_record_send_records_history_and_audit(monkeypatch):
    store = _setup_store(monkeypatch)
    events = []
    monkeypatch.setattr(
        planner, "emit_event", lambda e, p, **k: events.append((e, p))
    )

    session = {
        "session_id": "s1",
        "strategy": {"accounts": [{"account_id": "1", "action_tag": "dispute"}]},
    }

    planner.plan_next_step(session, ["dispute"], now=datetime(2024, 1, 1))
    planner.record_send(session, ["1"], now=datetime(2024, 1, 2))

    state = planner.load_state(store["s1"]["account_states"]["1"])
    assert len(state.history) == 1
    hist = state.history[0]
    assert hist.from_status == AccountStatus.PLANNED
    assert hist.to_status == AccountStatus.SENT
    assert hist.actor == "planner"
    assert events == [
        (
            "audit.planner_transition",
            {"account_id": "1", "cycle": 0, "step": 1, "reason": "letters_sent"},
        )
    ]


def test_account_transitions_endpoint_returns_history(monkeypatch):
    store = _setup_store(monkeypatch)

    session = {
        "session_id": "sess1",
        "strategy": {"accounts": [{"account_id": "1", "action_tag": "dispute"}]},
    }

    planner.plan_next_step(session, ["dispute"], now=datetime(2024, 1, 1))
    planner.record_send(session, ["1"], now=datetime(2024, 1, 2))

    monkeypatch.setattr(api_app, "get_session", lambda sid: store.get(sid))
    flask_app = api_app.create_app()
    resp = flask_app.test_client().get("/api/account-transitions/sess1/1")
    assert resp.status_code == 200
    data = resp.get_json()
    assert len(data["history"]) == 1
    assert data["history"][0]["actor"] == "planner"
