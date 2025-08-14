import json
from pathlib import Path
from unittest import mock

from backend.core.logic.strategy.generate_strategy_report import StrategyGenerator
from tests.helpers.fake_ai_client import FakeAIClient


def test_stage_2_5_data_included_in_prompt_and_saved(tmp_path):
    fake = FakeAIClient()
    # minimal valid strategy response
    fake.add_chat_response(
        json.dumps(
            {
                "overview": "",
                "accounts": [{"account_id": "1", "name": "Acc"}],
                "global_recommendations": [],
            }
        )
    )
    gen = StrategyGenerator(ai_client=fake)
    stage_2_5 = {
        "1": {
            "legal_safe_summary": "Safe summary",
            "suggested_dispute_frame": "fraud",
            "rule_hits": ["E_IDENTITY"],
            "needs_evidence": ["identity_theft_affidavit"],
            "red_flags": ["admission_of_fault"],
        }
    }
    with mock.patch(
        "backend.core.logic.strategy.generate_strategy_report.fix_draft_with_guardrails",
        lambda *a, **k: None,
    ):
        report = gen.generate(
            {"name": "Client", "session_id": "sess"},
            {"Experian": {}},
            stage_2_5_data=stage_2_5,
            audit=None,
        )
    prompt = fake.chat_payloads[0]["messages"][0]["content"]
    assert "legal_safe_summary" in prompt
    assert "Safe summary" in prompt
    path = gen.save_report(
        report,
        {"name": "Client", "session_id": "sess"},
        "2024-01-01",
        base_dir=tmp_path,
        stage_2_5_data=stage_2_5,
    )
    saved = json.loads(Path(path).read_text())
    assert saved["accounts"][0]["legal_safe_summary"] == "Safe summary"
    assert saved["accounts"][0]["rule_hits"] == ["E_IDENTITY"]
    assert saved["accounts"][0]["needs_evidence"] == ["identity_theft_affidavit"]
    assert saved["accounts"][0]["red_flags"] == ["admission_of_fault"]
