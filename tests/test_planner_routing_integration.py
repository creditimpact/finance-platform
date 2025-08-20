import planner
import tactical
from backend.api import session_manager
from backend.core.letters import router as letters_router


def test_candidate_routing_precedes_planner_and_finalize_after(monkeypatch):
    monkeypatch.setenv("LETTERS_ROUTER_PHASED", "1")

    # simple in-memory session store
    store = {}

    def fake_get_session(sid):
        return store.get(sid)

    def fake_update_session(sid, **kwargs):
        session = store.setdefault(sid, {})
        session.update(kwargs)
        return session

    monkeypatch.setattr(session_manager, "get_session", fake_get_session)
    monkeypatch.setattr(session_manager, "update_session", fake_update_session)

    order = []

    orig_select = letters_router.select_template

    def wrapped_select(tag, ctx, phase, session_id=None):
        order.append(f"{phase}:{tag}")
        return orig_select(tag, ctx, phase, session_id=session_id)

    monkeypatch.setattr(letters_router, "select_template", wrapped_select)

    # Patch core letter generation to invoke finalize routing for allowed tags
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
            letters_router.select_template(acc.get("action_tag"), acc, phase="finalize")
        return []

    monkeypatch.setattr(core_orch, "generate_letters", fake_generate_letters)

    orig_plan = planner.plan_next_step

    def wrapped_plan(session, action_tags):
        order.append("planner")
        return orig_plan(session, action_tags)

    monkeypatch.setattr(planner, "plan_next_step", wrapped_plan)

    stage_2_5 = {
        "1": {
            "action_tag": "goodwill"
        },  # missing creditor to trigger candidate warnings
        "2": {"action_tag": "dispute", "bureau": "Experian"},
    }

    strategy = {
        "accounts": [
            {"account_id": "1", "action_tag": "goodwill"},
            {"account_id": "2", "action_tag": "dispute", "bureau": "Experian"},
        ]
    }

    session_ctx = {"session_id": "sess1", "strategy": strategy}

    decisions = {
        acc_id: letters_router.select_template(
            ctx["action_tag"], ctx, phase="candidate"
        )
        for acc_id, ctx in stage_2_5.items()
    }

    allowed_tags = planner.plan_next_step(
        session_ctx, [ctx["action_tag"] for ctx in stage_2_5.values()]
    )
    assert allowed_tags == ["dispute"]
    tactical.generate_letters(session_ctx, allowed_tags)

    assert order == [
        "candidate:goodwill",
        "candidate:dispute",
        "planner",
        "finalize:dispute",
    ]
    assert decisions["1"].required_fields == ["creditor"]
