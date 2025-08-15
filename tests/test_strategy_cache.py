import json

from backend.core.logic.strategy.generate_strategy_report import (
    StrategyGenerator,
    STRATEGY_PROMPT_VERSION,
)
from backend.core.cache import strategy_cache
from tests.helpers.fake_ai_client import FakeAIClient


def test_strategy_cache_hit_and_version_bump(monkeypatch):
    monkeypatch.setattr(
        "backend.core.logic.strategy.generate_strategy_report.fix_draft_with_guardrails",
        lambda *a, **k: None,
    )
    strategy_cache.reset_cache()
    fake = FakeAIClient()
    fake.add_chat_response(json.dumps({"overview": "", "accounts": [], "global_recommendations": []}))
    gen = StrategyGenerator(fake)
    bureau_data = {"Experian": {"disputes": []}}
    stage_2_5 = {}
    classification_map = {}

    res1 = gen.generate({}, bureau_data, classification_map=classification_map, stage_2_5_data=stage_2_5)
    assert res1["accounts"] == []
    res2 = gen.generate({}, bureau_data, classification_map=classification_map, stage_2_5_data=stage_2_5)
    assert res1 == res2
    assert len(fake.chat_payloads) == 1

    monkeypatch.setattr(
        "backend.core.logic.strategy.generate_strategy_report.STRATEGY_PROMPT_VERSION",
        STRATEGY_PROMPT_VERSION + 1,
    )
    fake.add_chat_response(json.dumps({"overview": "y", "accounts": [], "global_recommendations": []}))
    res3 = gen.generate({}, bureau_data, classification_map=classification_map, stage_2_5_data=stage_2_5)
    assert len(fake.chat_payloads) == 2
    assert res3["overview"] == "y"
