import os

from backend.analytics.analytics_tracker import get_counters, reset_counters
from backend.core.letters.router import select_template
from backend.core.logic.strategy.normalizer_2_5 import evaluate_rules
from backend.policy.policy_loader import load_rulebook


def test_candidate_routing_emits_missing_fields_after_stage_2_5(monkeypatch):
    monkeypatch.setenv("LETTERS_ROUTER_PHASED", "1")
    reset_counters()

    tri_merge = {
        "mismatch_types": ["presence"],
        "family_id": "fam1",
        "evidence_snapshot_id": "snap1",
    }
    ctx = evaluate_rules("", {}, load_rulebook(), tri_merge=tri_merge)

    decision = select_template(ctx["action_tag"], ctx, phase="candidate")
    assert set(decision.missing_fields) == {
        "creditor_name",
        "account_number_masked",
        "bureau",
        "legal_safe_summary",
    }

    counters = get_counters()
    tag = ctx["action_tag"]
    template = "bureau_dispute_letter_template.html"
    assert counters.get("router.candidate_selected") == 1
    assert counters.get(f"router.candidate_selected.{tag}") == 1
    for field in decision.missing_fields:
        key = f"router.missing_fields.{tag}.{template}.{field}"
        assert counters.get(key) == 1
