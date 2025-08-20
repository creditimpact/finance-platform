import pytest

from backend.core.logic.report_analysis.tri_merge_models import (
    Mismatch,
    TradelineFamily,
)
from backend.core.logic.strategy.normalizer_2_5 import evaluate_rules
from backend.policy.policy_loader import load_rulebook


@pytest.mark.parametrize(
    "mismatch_type,expected_rule,expected_tag",
    [
        ("presence", "TM_PRESENCE", "bureau_dispute"),
        ("balance", "TM_BALANCE", "mov"),
        ("status", "TM_STATUS", "mov"),
        ("dates", "TM_DATES", "mov"),
        ("remarks", "TM_REMARKS", "bureau_dispute"),
        ("utilization", "TM_UTILIZATION", "mov"),
        ("personal_info", "TM_PERSONAL_INFO", "personal_info_correction"),
        ("duplicate", "TM_DUPLICATE", "bureau_dispute"),
    ],
)
def test_tri_merge_rules_produce_canonical_tags_and_evidence(
    mismatch_type: str, expected_rule: str, expected_tag: str
) -> None:
    rulebook = load_rulebook()
    fam = TradelineFamily(account_number="1234")
    fam.mismatches.append(Mismatch(field=mismatch_type, values={}))
    family_id = "fam123"
    setattr(fam, "family_id", family_id)
    tri_info = {
        "family_id": family_id,
        "mismatch_types": [m.field for m in fam.mismatches],
        "evidence_snapshot_id": family_id,
    }
    result = evaluate_rules("", {}, rulebook, tri_info)
    assert result["rule_hits"] == [expected_rule]
    assert result["action_tag"] == expected_tag
    assert result["needs_evidence"] == ["tri_merge_snapshot"]
    assert result["tri_merge"]["evidence_snapshot_id"] == family_id


@pytest.mark.parametrize(
    "mismatch_type,expected_rule,expected_tag",
    [
        ("presence", "TM_PRESENCE", "bureau_dispute"),
        ("balance", "TM_BALANCE", "mov"),
        ("status", "TM_STATUS", "mov"),
        ("dates", "TM_DATES", "mov"),
        ("remarks", "TM_REMARKS", "bureau_dispute"),
        ("utilization", "TM_UTILIZATION", "mov"),
        ("personal_info", "TM_PERSONAL_INFO", "personal_info_correction"),
        ("duplicate", "TM_DUPLICATE", "bureau_dispute"),
    ],
)
def test_tri_merge_rules_supersede_inaccuracy_disputes(
    mismatch_type: str, expected_rule: str, expected_tag: str
) -> None:
    rulebook = load_rulebook()
    facts = {
        "is_inaccurate_or_incomplete": True,
        "has_direct_dispute_address": True,
    }
    fam = TradelineFamily(account_number="1234")
    fam.mismatches.append(Mismatch(field=mismatch_type, values={}))
    family_id = "fam123"
    setattr(fam, "family_id", family_id)
    tri_info = {
        "family_id": family_id,
        "mismatch_types": [m.field for m in fam.mismatches],
        "evidence_snapshot_id": family_id,
    }
    result = evaluate_rules("", facts, rulebook, tri_info)
    assert result["rule_hits"] == [expected_rule]
    assert result["action_tag"] == expected_tag
