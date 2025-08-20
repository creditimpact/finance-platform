import pytest

from backend.analytics.analytics_tracker import get_counters, reset_counters
from backend.core.letters.router import select_template
from backend.core.logic.strategy.normalizer_2_5 import evaluate_rules
from backend.policy.policy_loader import load_rulebook

MOV_FIELDS = {
    "creditor_name",
    "account_number_masked",
    "legal_safe_summary",
    "cra_last_result",
    "days_since_cra_result",
    "reinvestigation_request",
    "method_of_verification",
}

BUREAU_FIELDS = {
    "creditor_name",
    "account_number_masked",
    "bureau",
    "legal_safe_summary",
    "fcra_611",
    "reinvestigation_request",
}

PII_FIELDS = {
    "client_name",
    "client_address_lines",
    "date_of_birth",
    "ssn_last4",
    "legal_safe_summary",
}


@pytest.mark.parametrize(
    "mismatch_rule,template,fields",
    [
        ("TM_BALANCE", "mov_letter_template.html", MOV_FIELDS),
        ("TM_STATUS", "mov_letter_template.html", MOV_FIELDS),
        ("TM_DATES", "mov_letter_template.html", MOV_FIELDS),
        ("TM_REMARKS", "bureau_dispute_letter_template.html", BUREAU_FIELDS),
        ("TM_UTILIZATION", "mov_letter_template.html", MOV_FIELDS),
        (
            "TM_PERSONAL_INFO",
            "personal_info_correction_letter_template.html",
            PII_FIELDS,
        ),
        ("TM_DUPLICATE", "bureau_dispute_letter_template.html", BUREAU_FIELDS),
    ],
)
def test_finalize_routing_emits_missing_fields_after_stage_2_5(
    monkeypatch, mismatch_rule: str, template: str, fields: set[str]
):
    monkeypatch.setenv("LETTERS_ROUTER_PHASED", "1")
    reset_counters()
    mismatch_type = mismatch_rule.removeprefix("TM_").lower()

    tri_merge = {
        "mismatch_types": [mismatch_type],
        "family_id": "fam1",
        "evidence_snapshot_id": "snap1",
    }
    ctx = evaluate_rules("", {}, load_rulebook(), tri_merge=tri_merge)

    decision = select_template(ctx["action_tag"], ctx, phase="finalize")
    assert decision.template_path == template
    assert set(decision.missing_fields) == fields

    counters = get_counters()
    tag = ctx["action_tag"]
    assert counters.get("router.finalized") == 1
    assert counters.get(f"router.finalized.{tag}") == 1
    for field in decision.missing_fields:
        key = f"router.missing_fields.{tag}.{template}.{field}"
        assert counters.get(key) == 1
