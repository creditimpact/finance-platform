import pytest

from backend.core.logic.strategy.generate_strategy_report import StrategyGenerator
from tests.helpers.fake_ai_client import FakeAIClient


def test_save_report_fails_without_required_fields(tmp_path):
    fake = FakeAIClient()
    gen = StrategyGenerator(ai_client=fake)
    report = {"overview": "", "accounts": [{"account_id": "1"}], "global_recommendations": []}
    with pytest.raises(ValueError):
        gen.save_report(
            report,
            {"name": "Client", "session_id": "sess"},
            "2024-01-01",
            base_dir=tmp_path,
            stage_2_5_data={"1": {}}
        )
