import json
from pathlib import Path

import pytest

from backend.analytics.analytics_tracker import (
    emit_counter,
    get_counters,
    reset_counters,
)
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
    reset_counters()
    rulebook = load_rulebook()
    tri_info = {
        "family_id": "fam123",
        "mismatch_types": [mismatch_type],
        "evidence_snapshot_id": "fam123",
    }
    facts = {
        "is_inaccurate_or_incomplete": True,
        "has_direct_dispute_address": True,
    }
    result = evaluate_rules("", facts, rulebook, tri_info)
    emit_counter(f"rulebook.tag_selected.{result['action_tag']}")
    assert result["rule_hits"] == [expected_rule]
    assert result["action_tag"] == expected_tag
    counters = get_counters()
    assert counters[f"rulebook.tag_selected.{expected_tag}"] == 1
    assert counters["rulebook.suppressed_rules.A_CRA_DISPUTE"] == 1
    assert counters["rulebook.suppressed_rules.B_DIRECT_DISPUTE"] == 1


def test_conflicting_mismatches_follow_precedence() -> None:
    """Presence and balance mismatches should favor presence tag."""
    reset_counters()
    rulebook = load_rulebook()
    tri_info = {
        "family_id": "fam123",
        "mismatch_types": ["presence", "balance"],
        "evidence_snapshot_id": "fam123",
    }
    result = evaluate_rules("", {}, rulebook, tri_info)
    emit_counter(f"rulebook.tag_selected.{result['action_tag']}")
    assert result["rule_hits"] == ["TM_PRESENCE", "TM_BALANCE"]
    assert result["action_tag"] == "bureau_dispute"
    counters = get_counters()
    assert counters["rulebook.tag_selected.bureau_dispute"] == 1
    assert "rulebook.suppressed_rules.TM_BALANCE" not in counters


def test_pii_only_inputs_yield_personal_info_correction() -> None:
    reset_counters()
    rulebook = load_rulebook()
    tri_info = {
        "family_id": "fam123",
        "mismatch_types": ["personal_info"],
        "evidence_snapshot_id": "fam123",
    }
    result = evaluate_rules("", {}, rulebook, tri_info)
    emit_counter(f"rulebook.tag_selected.{result['action_tag']}")
    assert result["rule_hits"] == ["TM_PERSONAL_INFO"]
    assert result["action_tag"] == "personal_info_correction"
    counters = get_counters()
    assert counters["rulebook.tag_selected.personal_info_correction"] == 1


def test_golden_conflicting_mismatches_consistent_tags() -> None:
    rulebook = load_rulebook()
    tri_info = {
        "family_id": "fam123",
        "mismatch_types": ["presence", "balance"],
        "evidence_snapshot_id": "fam123",
    }
    res1 = evaluate_rules("", {}, rulebook, tri_info)
    res2 = evaluate_rules("", {}, rulebook, tri_info)
    assert res1 == res2
    golden_path = Path("tests/strategy/goldens/tri_merge_presence_balance.json")
    expected = json.loads(golden_path.read_text())
    assert res1["rule_hits"] == expected["rule_hits"]
    assert res1["action_tag"] == expected["action_tag"]
