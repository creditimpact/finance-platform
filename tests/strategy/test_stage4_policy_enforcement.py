import json

import pytest

from backend.core.logic.strategy.generate_strategy_report import StrategyGenerator
from tests.helpers.fake_ai_client import FakeAIClient
from backend.core.logic.policy import precedence_version


@pytest.fixture
def strategy_generator(monkeypatch) -> StrategyGenerator:
    """Strategy generator with AI client returning policy-violating output."""
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
                        "recommendation": "Send goodwill letter.",
                        "alternative_options": [],
                        "flags": [],
                        "legal_safe_summary": "",
                        "suggested_dispute_frame": "",
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
                        "needs_evidence": [],
                        "red_flags": [],
                    },
                ],
                "global_recommendations": [],
            }
        )
    )
    monkeypatch.setattr(
        "backend.core.logic.strategy.generate_strategy_report.fix_draft_with_guardrails",
        lambda *_, **__: None,
    )
    return StrategyGenerator(ai_client=fake)


@pytest.fixture
def stage_4_output(strategy_generator: StrategyGenerator) -> dict:
    stage_2_5_data = {
        "1": {"rule_hits": ["no_goodwill_on_collections"], "precedence_version": precedence_version},
        "2": {"rule_hits": ["fraud_flow"], "precedence_version": precedence_version},
    }
    return strategy_generator.generate({}, {}, stage_2_5_data=stage_2_5_data)


def test_policy_enforcement_applies_overrides(stage_4_output: dict) -> None:
    """Stage 4 should enforce policy restrictions from earlier stages."""
    acc1 = next(a for a in stage_4_output["accounts"] if a["account_id"] == "1")
    acc2 = next(a for a in stage_4_output["accounts"] if a["account_id"] == "2")

    assert acc1["recommendation"] == "Dispute with bureau"
    assert acc1["policy_override"] is True
    assert acc1["enforced_rules"] == ["no_goodwill_on_collections"]
    assert acc1["rule_hits"] == ["no_goodwill_on_collections"]
    assert acc1["precedence_version"] == precedence_version

    assert acc2["recommendation"] == "Fraud dispute"
    assert acc2["policy_override"] is True
    assert acc2["enforced_rules"] == ["fraud_flow"]
    assert acc2["rule_hits"] == ["fraud_flow"]
    assert acc2["precedence_version"] == precedence_version
