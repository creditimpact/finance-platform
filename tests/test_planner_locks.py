from contextlib import contextmanager
from datetime import datetime, timedelta

import planner
from backend.api import session_manager
from backend.outcomes import OutcomeEvent


def _setup(monkeypatch):
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


def test_plan_next_step_uses_account_lock(monkeypatch):
    _setup(monkeypatch)
    session = {
        "session_id": "s1",
        "strategy": {"accounts": [{"account_id": "1", "action_tag": "dispute"}]},
    }

    calls = []

    @contextmanager
    def fake_lock(acc_id):
        calls.append(acc_id)
        yield

    monkeypatch.setattr(planner, "account_lock", fake_lock)
    allowed = planner.plan_next_step(session, ["dispute"], now=datetime(2024, 1, 1))
    assert allowed == ["dispute"]
    assert calls == ["1"]


def test_handle_outcome_nochange_sets_next_eligible_and_uses_lock(monkeypatch):
    store = _setup(monkeypatch)
    session = {
        "session_id": "s1",
        "strategy": {"accounts": [{"account_id": "1", "action_tag": "dispute"}]},
    }

    send_time = datetime(2024, 1, 1)
    planner.plan_next_step(session, ["dispute"], now=send_time)
    planner.record_send(session, ["1"], now=send_time, sla_days=30)

    calls = []

    @contextmanager
    def fake_lock(acc_id):
        calls.append(acc_id)
        yield

    monkeypatch.setattr(planner, "account_lock", fake_lock)

    event = OutcomeEvent(
        outcome_id="o1",
        account_id="1",
        cycle_id=0,
        family_id="f1",
        outcome="NoChange",
    )
    allowed = planner.handle_outcome(
        session, event, now=send_time + timedelta(days=5), sla_days=30
    )
    assert allowed == []
    state = planner.load_state(store["s1"]["account_states"]["1"])
    assert state.next_eligible_at == send_time + timedelta(days=30)
    assert calls == ["1"]
