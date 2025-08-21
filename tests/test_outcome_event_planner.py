from datetime import datetime

import planner
from backend.api import session_manager
from backend.core.models import AccountStatus
from backend.outcomes import OutcomeEvent, load_outcome_history
from services import outcome_ingestion


def test_ingested_outcome_updates_planner(monkeypatch):
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

    event = OutcomeEvent(
        outcome_id="o1",
        account_id="1",
        cycle_id=0,
        family_id="f1",
        outcome="Verified",
    )
    outcome_ingestion.ingest(session, event)

    history = load_outcome_history("s1", "1")
    assert history == [event]

    allowed = planner.plan_next_step(session, ["followup"], now=datetime(2024, 3, 5))
    assert allowed == []
    final_state = planner.load_state(store["s1"]["account_states"]["1"])
    assert final_state.status == AccountStatus.COMPLETED
    assert final_state.last_outcome == "Verified"
    assert final_state.resolution_cycle_count == 1
    assert len(final_state.outcome_history) == 1
