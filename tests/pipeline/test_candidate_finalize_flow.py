import json
from pathlib import Path

from backend.analytics.analytics_tracker import (
    get_counters,
    get_missing_fields_heatmap,
    reset_counters,
)
from backend.core.letters.router import select_template
from backend.core.logic.letters.utils import populate_required_fields


def test_candidate_finalize_flow_golden(monkeypatch):
    monkeypatch.setenv("LETTERS_ROUTER_PHASED", "1")
    reset_counters()

    ctx = {
        "account_number_masked": "****1234",
        "legal_safe_summary": "Summary",
        "client": {"full_name": "Jane", "address_line": "1 St"},
        "today": "2024-01-01",
    }

    pre_decision = select_template("pay_for_delete", ctx, phase="finalize")
    pre_heatmap = get_missing_fields_heatmap()

    acc = {"action_tag": "pay_for_delete", "name": "ABC Collections"}
    strat = {"offer_terms": "Pay to delete"}
    populate_required_fields(acc, strat)
    ctx.update({"collector_name": acc["collector_name"], "offer_terms": acc["offer_terms"]})

    final_decision = select_template("pay_for_delete", ctx, phase="finalize")

    assert len(final_decision.missing_fields) < len(pre_decision.missing_fields)

    result = {
        "initial_template": pre_decision.template_path,
        "final_template": final_decision.template_path,
        "initial_missing_fields": sorted(pre_decision.missing_fields),
        "final_missing_fields": final_decision.missing_fields,
        "initial_missing_heatmap": pre_heatmap,
        "counters": get_counters(),
    }

    golden_path = Path("tests/pipeline/goldens/candidate_finalize_flow.json")
    expected = json.loads(golden_path.read_text())
    assert result == expected
