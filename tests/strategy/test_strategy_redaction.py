from pathlib import Path

from backend.audit.audit import create_audit_logger
from backend.core.logic.strategy.generate_strategy_report import StrategyGenerator
from tests.helpers.fake_ai_client import FakeAIClient


def test_strategy_and_audit_mask_account_numbers_and_ssn(tmp_path: Path) -> None:
    report = {
        "overview": "",
        "accounts": [
            {
                "account_id": "1",
                "name": "Bank",
                "account_number": "0000111122223333",
                "ssn": "123-45-6789",
                "recommendation": "Dispute",
            }
        ],
        "global_recommendations": [],
    }
    client = {"name": "Jane Doe", "session_id": "sess1"}

    gen = StrategyGenerator(ai_client=FakeAIClient())
    strategy_path = gen.save_report(
        report, client, "2024-01-01", base_dir=str(tmp_path)
    )
    text = strategy_path.read_text()
    assert "0000111122223333" not in text
    assert "123-45-6789" not in text
    assert "****3333" in text
    assert "***-**-6789" in text

    audit = create_audit_logger("sess1")
    audit.log_step("strategy_generated", report)
    audit_path = audit.save(tmp_path)
    audit_text = audit_path.read_text()
    assert "0000111122223333" not in audit_text
    assert "123-45-6789" not in audit_text
    assert "****3333" in audit_text
    assert "***-**-6789" in audit_text
