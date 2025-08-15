import json

import pytest

from backend.core.logic.strategy.generate_strategy_report import StrategyGenerator
from backend.core.logic.policy import precedence_version
from tests.helpers.fake_ai_client import FakeAIClient


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
    monkeypatch.setattr(
        "backend.core.logic.strategy.generate_strategy_report.get_cached_strategy",
        lambda *a, **k: None,
    )
    monkeypatch.setattr(
        "backend.core.logic.strategy.generate_strategy_report.store_cached_strategy",
        lambda *a, **k: None,
    )
    return StrategyGenerator(ai_client=fake)


def _stage2_5() -> dict:
    return {
        "1": {"rule_hits": ["no_goodwill_on_collections"], "precedence_version": precedence_version},
        "2": {"rule_hits": ["fraud_flow"], "precedence_version": precedence_version},
    }


def test_policy_enforcement_applies_overrides(monkeypatch, strategy_generator) -> None:
    monkeypatch.setenv("STAGE4_POLICY_ENFORCEMENT", "1")
    monkeypatch.setattr("backend.api.config.STAGE4_POLICY_ENFORCEMENT", True)
    monkeypatch.setattr(
        "backend.core.logic.strategy.generate_strategy_report.STAGE4_POLICY_ENFORCEMENT",
        True,
    )
    output = strategy_generator.generate({}, {}, stage_2_5_data=_stage2_5())
    acc1 = next(a for a in output["accounts"] if a["account_id"] == "1")
    acc2 = next(a for a in output["accounts"] if a["account_id"] == "2")

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


def test_shadow_mode_logs(monkeypatch, strategy_generator) -> None:
    monkeypatch.delenv("STAGE4_POLICY_ENFORCEMENT", raising=False)
    monkeypatch.setattr("backend.api.config.STAGE4_POLICY_ENFORCEMENT", False)
    monkeypatch.setattr("backend.api.config.STAGE4_POLICY_CANARY", 0.0)
    monkeypatch.setattr(
        "backend.core.logic.strategy.generate_strategy_report.STAGE4_POLICY_ENFORCEMENT",
        False,
    )
    monkeypatch.setattr(
        "backend.core.logic.strategy.generate_strategy_report.STAGE4_POLICY_CANARY",
        0.0,
    )
    events: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        "backend.core.logic.strategy.generate_strategy_report.emit_event",
        lambda e, p: events.append((e, p)),
    )
    output = strategy_generator.generate({}, {}, stage_2_5_data=_stage2_5())
    acc1 = next(a for a in output["accounts"] if a["account_id"] == "1")
    acc2 = next(a for a in output["accounts"] if a["account_id"] == "2")

    assert acc1["recommendation"] == "Send goodwill letter."
    assert acc1.get("policy_override") is not True
    assert acc1.get("enforced_rules") in (None, [])
    assert acc2["recommendation"] == "Dispute with bureau"
    assert acc2.get("policy_override") is not True
    assert acc2.get("enforced_rules") in (None, [])

    shadows = [payload for event, payload in events if event == "strategy_rule_enforcement"]
    assert all(p.get("shadow") for p in shadows)
    assert shadows[0]["would_apply"] == ["no_goodwill_on_collections"]
    assert shadows[1]["would_apply"] == ["fraud_flow"]
