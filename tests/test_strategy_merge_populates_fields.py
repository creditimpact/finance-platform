import os

from backend.analytics.analytics_tracker import (
    get_missing_fields_heatmap,
    reset_counters,
)
from backend.core.letters.router import select_template
from backend.core.logic.letters.utils import populate_required_fields


def test_strategy_merge_resolves_missing_fields(monkeypatch):
    monkeypatch.setenv("LETTERS_ROUTER_PHASED", "1")
    reset_counters()

    # After normalization/tagging - fields missing
    ctx = {
        "account_number_masked": "****1234",
        "legal_safe_summary": "Summary",
    }
    decision = select_template("pay_for_delete", ctx, phase="finalize")
    assert set(decision.missing_fields) == {
        "collector_name",
        "offer_terms",
        "deletion_clause",
        "payment_clause",
    }

    # Strategy merge fills in required fields
    acc = {"action_tag": "pay_for_delete", "name": "ABC Collections"}
    strat = {"offer_terms": "Pay to delete"}
    populate_required_fields(acc, strat)
    ctx.update({
        "collector_name": acc["collector_name"],
        "offer_terms": acc["offer_terms"],
    })

    decision = select_template("pay_for_delete", ctx, phase="finalize")
    assert decision.missing_fields == []

    heatmap = get_missing_fields_heatmap()
    assert heatmap["pay_for_delete"]["collector_name"] == 1
    assert heatmap["pay_for_delete"]["offer_terms"] == 1
