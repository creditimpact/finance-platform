import json


from backend.core.logic.strategy.generate_strategy_report import StrategyGenerator
from tests.helpers.fake_ai_client import FakeAIClient


def test_policy_based_overrides(monkeypatch):
    monkeypatch.setattr(
        "backend.core.logic.strategy.generate_strategy_report.fix_draft_with_guardrails",
        lambda *a, **k: None,
    )

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
                        "rule_hits": [],
                        "needs_evidence": [],
                        "red_flags": [],
                    },
                    {
                        "account_id": "2",
                        "name": "B",
                        "account_number": "2",
                        "status": "",
                        "analysis": "",
                        "recommendation": "Dispute with bureau",
                        "alternative_options": [],
                        "flags": [],
                        "legal_safe_summary": "",
                        "suggested_dispute_frame": "",
                        "rule_hits": [],
                        "needs_evidence": [],
                        "red_flags": [],
                    },
                ],
                "global_recommendations": [],
            }
        )
    )

    generator = StrategyGenerator(ai_client=fake)
    stage_2_5_data = {
        "1": {"rule_hits": ["no_goodwill_on_collections"]},
        "2": {"rule_hits": ["fraud_flow"]},
    }
    result = generator.generate({}, {}, stage_2_5_data=stage_2_5_data)

    acc1 = next(a for a in result["accounts"] if a["account_id"] == "1")
    acc2 = next(a for a in result["accounts"] if a["account_id"] == "2")

    assert acc1["recommendation"] == "Dispute with bureau"
    assert acc1["policy_override"] is True
    assert acc1["enforced_rules"] == ["no_goodwill_on_collections"]
    assert "no_goodwill_on_collections" in acc1["policy_override_reason"]

    assert acc2["recommendation"] == "Fraud dispute"
    assert acc2["policy_override"] is True
    assert acc2["enforced_rules"] == ["fraud_flow"]
    assert "fraud_flow" in acc2["policy_override_reason"]

