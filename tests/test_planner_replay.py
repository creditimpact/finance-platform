import copy
from datetime import datetime

import planner
from backend.api import session_manager


def test_planner_replay_produces_same_results(monkeypatch):
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

    allowed1 = planner.plan_next_step(session, ["dispute"], now=datetime(2024, 1, 1))
    snapshot = copy.deepcopy(store)
    allowed2 = planner.plan_next_step(session, ["dispute"], now=datetime(2024, 1, 1))

    assert allowed1 == allowed2 == ["dispute"]
    assert store == snapshot
