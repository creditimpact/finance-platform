import pytest

from backend.core.letters.field_population import apply_field_fillers
from backend.core.letters.router import select_template
from backend.analytics.analytics_tracker import reset_counters, get_counters

SCENARIOS = [
    (
        "pay_for_delete",
        {
            "action_tag": "pay_for_delete",
            "legal_safe_summary": "I will pay if you delete this account.",
            "tri_merge": {
                "name": "ABC Collections",
                "account_number_masked": "****1234",
            },
        },
        {"offer_terms": "Delete and pay 60%"},
        {"name", "creditor_name", "account_number_masked"},
    ),
    (
        "inquiry_dispute",
        {
            "action_tag": "inquiry_dispute",
            "bureau": "Experian",
            "legal_safe_summary": "This inquiry was unauthorized.",
            "tri_merge": {"account_number_masked": "****1111"},
            "inquiry_evidence": {"name": "Inq Co", "date": "2024-01-01"},
        },
        None,
        {"account_number_masked", "inquiry_creditor_name", "inquiry_date"},
    ),
    (
        "medical_dispute",
        {
            "action_tag": "medical_dispute",
            "bureau": "Experian",
            "legal_safe_summary": "Paid medical debt under $500 should be removed.",
            "tri_merge": {"name": "Med Co", "account_number_masked": "****2222"},
            "medical_evidence": {"amount": 100, "status": "Unpaid"},
        },
        None,
        {
            "name",
            "creditor_name",
            "account_number_masked",
            "amount",
            "medical_status",
        },
    ),
]


@pytest.mark.parametrize("tag, ctx, strat, expected", SCENARIOS)
def test_field_population_pipeline(monkeypatch, tag, ctx, strat, expected):
    reset_counters()
    monkeypatch.setenv("LETTERS_ROUTER_PHASED", "1")
    if strat:
        apply_field_fillers(ctx, strategy=strat)
    else:
        apply_field_fillers(ctx)
    counters = get_counters()
    for field in expected:
        assert counters.get(f"fields.populated_total.field.{field}") == 1
    decision = select_template(tag, ctx, phase="finalize")
    assert decision.missing_fields == []
