from datetime import datetime

import planner
from backend.api import session_manager
from backend.core.models import AccountStatus


def test_outcome_ingestion_marks_account_completed(monkeypatch):
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

    planner.plan_next_step(session, ["dispute"], now=datetime(2024, 1, 1))
    planner.record_send(session, ["1"], now=datetime(2024, 1, 2))

    # simulate CRA response ingestion
    state = planner.load_state(store["s1"]["account_states"]["1"])
    state.transition(AccountStatus.CRA_RESPONDED_VERIFIED, actor="cra")
    state.transition(AccountStatus.COMPLETED, actor="system")
    store["s1"]["account_states"]["1"] = planner.dump_state(state)

    allowed = planner.plan_next_step(session, ["followup"], now=datetime(2024, 3, 5))
    assert allowed == []
    final_state = planner.load_state(store["s1"]["account_states"]["1"])
    assert final_state.status == AccountStatus.COMPLETED
