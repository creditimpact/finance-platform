from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import planner
import tactical
from backend.api import session_manager
from backend.core.letters import router as letters_router


def test_planner_two_cycles_golden(monkeypatch):
    monkeypatch.setenv("LETTERS_ROUTER_PHASED", "1")
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

    order = []
    orig_select = letters_router.select_template

    def wrapped_select(tag, ctx, phase, session_id=None):
        order.append(f"{phase}:{tag}")
        return orig_select(tag, ctx, phase, session_id=session_id)

    monkeypatch.setattr(letters_router, "select_template", wrapped_select)

    send_times = [datetime(2024, 1, 1), datetime(2024, 2, 1)]
    orig_record_send = planner.record_send

    def fake_record_send(session, account_ids):
        now = send_times.pop(0)
        orig_record_send(session, account_ids, now=now, sla_days=30)

    monkeypatch.setattr(planner, "record_send", fake_record_send)
    orig_plan = planner.plan_next_step

    def wrapped_plan(session, action_tags, now=None):
        order.append("planner")
        return orig_plan(session, action_tags, now=now)

    monkeypatch.setattr(planner, "plan_next_step", wrapped_plan)

    import backend.core.orchestrators as core_orch

    def fake_generate_letters(
        client_info,
        bureau_data,
        sections,
        today_folder,
        is_identity_theft,
        strategy,
        audit,
        log_messages,
        classification_map,
        ai_client,
        app_config,
    ):
        for acc in strategy.get("accounts", []):
            tag = acc.get("action_tag")
            call_tag = "dispute" if tag == "followup" else tag
            letters_router.select_template(call_tag, acc, phase="finalize")
        return []

    monkeypatch.setattr(core_orch, "generate_letters", fake_generate_letters)

    session = {
        "session_id": "s1",
        "strategy": {"accounts": [{"account_id": "1", "action_tag": "dispute"}]},
    }

    stage_2_5 = {"1": {"action_tag": "dispute"}}
    letters_router.select_template("dispute", stage_2_5["1"], phase="candidate")

    allowed = planner.plan_next_step(session, ["dispute"], now=datetime(2024, 1, 1))
    tactical.generate_letters(session, allowed)

    allowed = planner.plan_next_step(session, ["followup"], now=datetime(2024, 1, 15))
    session["strategy"]["accounts"][0]["action_tag"] = "followup"
    allowed = planner.plan_next_step(session, ["followup"], now=datetime(2024, 2, 5))
    tactical.generate_letters(session, allowed)

    state = planner.load_state(store["s1"]["account_states"]["1"])
    result = {
        "order": order,
        "final_current_step": state.current_step,
        "next_eligible_at": state.next_eligible_at.isoformat(),
    }

    golden_path = Path("tests/pipeline/goldens/planner_two_cycles.json")
    expected = json.loads(golden_path.read_text())
    assert result == expected
