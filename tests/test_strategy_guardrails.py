import json
from backend.core.logic.strategy.generate_strategy_report import StrategyGenerator
from tests.helpers.fake_ai_client import FakeAIClient


def test_harsh_recommendation_softened_and_action_tag_preserved(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "backend.api.session_manager.SESSION_FILE", tmp_path / "sessions.json"
    )
    fake = FakeAIClient()
    initial = {
        "overview": "",
        "accounts": [
            {
                "account_id": "1",
                "name": "Acc",
                "account_number": "",
                "status": "",
                "analysis": "",
                "recommendation": "These crooks must fix this",
                "alternative_options": [],
                "flags": ["f1"],
                "action_tag": "dispute",
                "priority": "high",
            }
        ],
        "global_recommendations": [],
    }
    fake.add_chat_response(json.dumps(initial))
    gen = StrategyGenerator(ai_client=fake)
    report = gen.generate({"session_id": "sess", "state": "CA"}, {})
    acc = report["accounts"][0]
    assert "crooks" not in acc["recommendation"].lower()
    assert acc["action_tag"] == initial["accounts"][0]["action_tag"]
    assert acc["priority"] == initial["accounts"][0]["priority"]
    assert acc["flags"] == initial["accounts"][0]["flags"]
