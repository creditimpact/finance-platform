from backend.core.logic.strategy.normalizer_2_5 import evaluate_rules
from backend.policy.policy_loader import load_rulebook


def test_identity_theft_precedence_and_exclusion() -> None:
    rb = load_rulebook()
    facts = {
        "identity_theft": True,
        "has_id_theft_affidavit": False,
        "type": "collection",
        "days_since_first_contact": 10,
    }
    result = evaluate_rules("", facts, rb)
    assert result["rule_hits"] == ["E_IDENTITY", "E_IDENTITY_NEEDS_AFFIDAVIT"]
    assert result["needs_evidence"] == ["identity_theft_affidavit"]
    assert result["suggested_dispute_frame"] == "fraud"


def test_validation_excludes_pay_for_delete() -> None:
    rb = load_rulebook()
    facts = {
        "type": "collection",
        "days_since_first_contact": 10,
        "is_inaccurate_or_incomplete": False,
        "not_able_to_verify": False,
        "years_since_dofd": 1,
        "identity_theft": False,
    }
    result = evaluate_rules("", facts, rb)
    assert result["rule_hits"] == ["D_VALIDATION"]
    assert result["suggested_dispute_frame"] == "debt_validation"


def test_tri_merge_presence_rule() -> None:
    rb = {
        "rules": [
            {
                "id": "TM_PRESENCE",
                "when": {"field": "tri_merge.presence", "eq": True},
                "effect": {"rule_hits": ["TM_PRESENCE"]},
            }
        ],
        "precedence": ["TM_PRESENCE"],
    }
    tri = {"mismatch_types": ["presence"], "evidence_snapshot_id": "snap1"}
    result = evaluate_rules("", {}, rb, tri)
    assert result["rule_hits"] == ["TM_PRESENCE"]
    assert result["tri_merge"]["evidence_snapshot_id"] == "snap1"
