import planner
from backend.api import session_manager
from datetime import datetime, timedelta


def test_next_eligible_at_and_planner_skip(monkeypatch):
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

    session = {
        "session_id": "s1",
        "strategy": {"accounts": [{"account_id": "1", "action_tag": "dispute"}]},
    }

    allowed = planner.plan_next_step(session, ["dispute"], now=datetime(2024, 1, 1))
    assert allowed == ["dispute"]

    send_time = datetime(2024, 1, 2)
    planner.record_send(session, ["1"], now=send_time, sla_days=30)

    # Before SLA expires - planner should skip account
    allowed = planner.plan_next_step(session, ["followup"], now=datetime(2024, 1, 20))
    assert allowed == []

    state_data = store["s1"]["account_states"]["1"]
    state = planner.load_state(state_data)
    assert state.last_sent_at == send_time
    assert state.next_eligible_at == send_time + timedelta(days=30)

    # After SLA expires
    allowed = planner.plan_next_step(session, ["followup"], now=datetime(2024, 2, 5))
    assert allowed == ["followup"]
