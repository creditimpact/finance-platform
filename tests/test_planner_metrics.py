import planner
from datetime import datetime
from backend.api import session_manager
from backend.analytics.analytics_tracker import reset_counters, get_counters
from backend.core.models import AccountState, AccountStatus


def test_planner_metric_emissions(monkeypatch):
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

    reset_counters()

    # Prepopulate one completed account to exercise resolution gauges
    state = AccountState(
        account_id="1",
        current_cycle=2,
        current_step=0,
        status=AccountStatus.COMPLETED,
    )
    store["s1"] = {"account_states": {"1": planner.dump_state(state)}}

    session = {
        "session_id": "s1",
        "strategy": {"accounts": [{"account_id": "1"}, {"account_id": "2"}]},
    }

    allowed = planner.plan_next_step(session, ["dispute"], now=datetime(2024, 1, 1))
    assert allowed == ["dispute"]

    planner.record_send(session, ["2"], now=datetime(2024, 1, 1), sla_days=30)

    # Trigger time_to_next_step_ms by planning before SLA expires
    planner.plan_next_step(session, ["followup"], now=datetime(2024, 1, 15))

    # Second send after SLA window to count violation
    planner.record_send(session, ["2"], now=datetime(2024, 2, 15), sla_days=30)

    counters = get_counters()

    assert counters["planner.cycle_progress"] == 2
    assert counters["planner.cycle_progress.cycle.0"] == 2
    assert counters["planner.cycle_progress.step.1"] == 1
    assert counters["planner.cycle_progress.step.2"] == 1
    assert counters["planner.sla_violations_total"] == 1
    assert counters["planner.cycle_success_rate"] == 0.5
    assert counters["planner.avg_cycles_per_resolution"] == 2.0
    assert counters.get("planner.error_count", 0) == 0
    assert counters["planner.time_to_next_step_ms"] > 0
