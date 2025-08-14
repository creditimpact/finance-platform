import json

from backend.core.logic.policy import precedence_version
from backend.core.logic.strategy.normalizer_2_5 import evaluate_rules
from backend.core.logic.strategy.generate_strategy_report import StrategyGenerator
from tests.helpers.fake_ai_client import FakeAIClient


def test_conflicting_rules_resolve_consistently(monkeypatch):
    rulebook = {
        "rules": [
            {
                "id": "fraud_flow",
                "when": {"field": "identity_theft", "eq": True},
                "effect": {"rule_hits": ["fraud_flow"]},
            },
            {
                "id": "no_goodwill_on_collections",
                "when": {"field": "type", "eq": "collection"},
                "effect": {"rule_hits": ["no_goodwill_on_collections"]},
            },
        ],
        "precedence": ["fraud_flow", "no_goodwill_on_collections"],
        "version": "test",
    }
    facts = {"identity_theft": True, "type": "collection"}
    eval_res = evaluate_rules("", facts, rulebook)
    assert eval_res["rule_hits"] == ["fraud_flow", "no_goodwill_on_collections"]

    fake = FakeAIClient()
    fake.add_chat_response(
        json.dumps(
            {
                "overview": "",
                "accounts": [
                    {
                        "account_id": "1",
                        "name": "A",
                        "account_number": "1",
                        "status": "",
                        "analysis": "",
                        "recommendation": "Goodwill",
                        "alternative_options": [],
                        "flags": [],
                        "legal_safe_summary": "",
                        "suggested_dispute_frame": "",
                        "needs_evidence": [],
                        "red_flags": [],
                    }
                ],
                "global_recommendations": [],
            }
        )
    )
    monkeypatch.setattr(
        "backend.core.logic.strategy.generate_strategy_report.fix_draft_with_guardrails",
        lambda *_, **__: None,
    )
    monkeypatch.setattr(
        "backend.core.logic.strategy.generate_strategy_report.load_rulebook",
        lambda: rulebook,
    )
    gen = StrategyGenerator(ai_client=fake)

    stage_2_5_data = {
        "1": {
            "legal_safe_summary": "",
            "suggested_dispute_frame": "",
            "rule_hits": list(reversed(eval_res["rule_hits"])),
            "needs_evidence": [],
            "red_flags": [],
            "prohibited_admission_detected": False,
            "rulebook_version": "test",
            "precedence_version": precedence_version,
        }
    }

    result = gen.generate({}, {}, stage_2_5_data=stage_2_5_data)
    acc = result["accounts"][0]
    assert acc["recommendation"] == "Fraud dispute"
    assert acc["enforced_rules"] == ["fraud_flow"]
    assert acc["precedence_version"] == precedence_version
    assert acc["enforced_rules"][0] == eval_res["rule_hits"][0]
