import json

from backend.analytics.analytics_tracker import get_counters, reset_counters
from backend.audit.audit import create_audit_logger
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

    events = []
    monkeypatch.setattr(
        "backend.core.logic.strategy.generate_strategy_report.emit_event",
        lambda event, payload: events.append((event, payload)),
    )

    reset_counters()

    generator = StrategyGenerator(ai_client=fake)
    audit = create_audit_logger("test")
    stage_2_5_data = {
        "1": {"rule_hits": ["no_goodwill_on_collections"]},
        "2": {"rule_hits": ["fraud_flow"]},
    }
    result = generator.generate({}, {}, stage_2_5_data=stage_2_5_data, audit=audit)

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

    counters = get_counters()
    assert counters["strategy.rule_hit_total"] == 2
    assert counters["strategy.policy_override_total"] == 2

    acc_logs = audit.data["accounts"]
    assert acc_logs["1"][0]["rule_hits"] == ["no_goodwill_on_collections"]
    assert acc_logs["1"][0]["applied_rules"] == ["no_goodwill_on_collections"]
    assert acc_logs["1"][0]["policy_override"] is True

    assert events[0][0] == "strategy_rule_enforcement"
    assert events[0][1]["account_id"] == "1"
    assert events[1][1]["applied_rules"] == ["fraud_flow"]

