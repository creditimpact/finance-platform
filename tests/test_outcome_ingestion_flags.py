from backend.api import session_manager
from backend.outcomes import OutcomeEvent
from services import outcome_ingestion
import planner


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
    return store


def _make_event():
    return OutcomeEvent(
        outcome_id="o1",
        account_id="1",
        cycle_id=0,
        family_id="f1",
        outcome="Verified",
    )


def test_ingest_respects_enable_flag(monkeypatch):
    _setup_store(monkeypatch)
    calls = {"n": 0}

    def fake_handle(sess, ev):
        calls["n"] += 1

    monkeypatch.setattr(planner, "handle_outcome", fake_handle)

    session = {"session_id": "s1"}
    event = _make_event()

    monkeypatch.setenv("ENABLE_OUTCOME_INGESTION", "false")
    outcome_ingestion.ingest(session, event)
    assert calls["n"] == 0


def test_ingest_respects_canary(monkeypatch):
    _setup_store(monkeypatch)
    calls = {"n": 0}

    def fake_handle(sess, ev):
        calls["n"] += 1

    monkeypatch.setattr(planner, "handle_outcome", fake_handle)

    session = {"session_id": "s1"}
    event = _make_event()

    monkeypatch.setenv("OUTCOME_INGESTION_CANARY_PERCENT", "0")
    outcome_ingestion.ingest(session, event)
    assert calls["n"] == 0

    monkeypatch.setenv("OUTCOME_INGESTION_CANARY_PERCENT", "100")
    outcome_ingestion.ingest(session, event)
    assert calls["n"] == 1
