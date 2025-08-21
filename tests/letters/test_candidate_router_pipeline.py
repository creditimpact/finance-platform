import os

from backend.analytics.analytics_tracker import get_counters, reset_counters
from backend.core.letters.router import select_template
import planner
import tactical


def test_candidate_router_metrics_before_planner(monkeypatch):
    monkeypatch.setenv("LETTERS_ROUTER_PHASED", "1")
    reset_counters()

    ctx = {"legal_safe_summary": "text"}
    tag = "pay_for_delete"

    # Candidate routing
    select_template(tag, ctx, phase="candidate")
    counters = get_counters()
    assert counters.get("router.candidate_selected") == 1
    assert counters.get(f"router.candidate_selected.{tag}") == 1
    assert any(k.startswith(f"router.missing_fields.{tag}") for k in counters)

    calls = []

    def fake_plan(session, tags):
        calls.append("planner")
        # Candidate metrics should already be present
        c = get_counters()
        assert c.get("router.candidate_selected") == 1
        return tags

    def fake_generate(session, tags):
        calls.append("tactical")
        c = get_counters()
        assert c.get("router.candidate_selected") == 1

    monkeypatch.setattr(planner, "plan_next_step", fake_plan)
    monkeypatch.setattr(tactical, "generate_letters", fake_generate)

    planner.plan_next_step({}, [tag])
    tactical.generate_letters({}, [tag])

    # Finalization
    select_template(tag, ctx, phase="finalize")
    assert "planner" in calls and "tactical" in calls
