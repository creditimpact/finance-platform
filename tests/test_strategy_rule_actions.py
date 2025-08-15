import json
import pytest

from backend.core.logic.strategy.generate_strategy_report import StrategyGenerator
from backend.core.logic.policy import precedence_version
from tests.helpers.fake_ai_client import FakeAIClient


@pytest.fixture
def strategy_generator(monkeypatch):
    monkeypatch.setenv("STAGE4_POLICY_ENFORCEMENT", "1")
    monkeypatch.setattr("backend.api.config.STAGE4_POLICY_ENFORCEMENT", True)
    monkeypatch.setattr(
        "backend.core.logic.strategy.generate_strategy_report.STAGE4_POLICY_ENFORCEMENT",
        True,
    )
    monkeypatch.setattr(
        "backend.core.logic.strategy.generate_strategy_report.fix_draft_with_guardrails",
        lambda *a, **k: None,
    )
    monkeypatch.setattr(
        "backend.core.logic.strategy.generate_strategy_report.get_cached_strategy",
        lambda *a, **k: None,
    )
    monkeypatch.setattr(
        "backend.core.logic.strategy.generate_strategy_report.store_cached_strategy",
        lambda *a, **k: None,
    )
    fake = FakeAIClient()
    return fake, StrategyGenerator(ai_client=fake)


def _base_account():
    return {
        "account_id": "1",
        "name": "A",
        "account_number": "1",
        "status": "",
        "analysis": "",
        "recommendation": "Dispute with bureau",
        "alternative_options": [],
        "flags": [],
        "legal_safe_summary": "",
        "suggested_dispute_frame": "",
        "needs_evidence": [],
        "red_flags": [],
    }


def test_paydown_first_required_action(strategy_generator):
    fake, generator = strategy_generator
    fake.add_chat_response(
        json.dumps({"overview": "", "accounts": [_base_account()], "global_recommendations": []})
    )
    stage_2_5_data = {"1": {"rule_hits": ["paydown_first"], "precedence_version": precedence_version}}
    result = generator.generate({}, {}, stage_2_5_data=stage_2_5_data)
    acc = result["accounts"][0]
    assert acc["recommendation"] == "Pay down before disputing"
    assert acc["policy_override"] is True
    assert acc["enforced_rules"] == ["paydown_first"]
    assert acc["required_actions"] == ["Pay down before disputing"]


def test_duplicate_tradeline_forbidden_action(strategy_generator):
    fake, generator = strategy_generator
    fake.add_chat_response(
        json.dumps({"overview": "", "accounts": [_base_account()], "global_recommendations": []})
    )
    stage_2_5_data = {"1": {"rule_hits": ["duplicate_tradeline"], "precedence_version": precedence_version}}
    result = generator.generate({}, {}, stage_2_5_data=stage_2_5_data)
    acc = result["accounts"][0]
    assert acc["recommendation"] == "Dispute with bureau"
    assert acc["policy_override"] is True
    assert acc["enforced_rules"] == ["duplicate_tradeline"]
    assert acc["forbidden_actions"] == ["Dispute with bureau"]


def test_unauthorized_inquiry_sets_flag(strategy_generator):
    fake, generator = strategy_generator
    fake.add_chat_response(
        json.dumps({"overview": "", "accounts": [_base_account()], "global_recommendations": []})
    )
    stage_2_5_data = {"1": {"rule_hits": ["unauthorized_inquiry"], "precedence_version": precedence_version}}
    result = generator.generate({}, {}, stage_2_5_data=stage_2_5_data)
    acc = result["accounts"][0]
    assert acc["flags"] == ["unauthorized_inquiry"]
    assert acc.get("policy_override") is None
    assert acc.get("required_actions", []) == []
    assert acc.get("forbidden_actions", []) == []
