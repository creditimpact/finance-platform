import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from audit import create_audit_logger
from logic.generate_strategy_report import StrategyGenerator
from logic.constants import StrategistFailureReason
from tests.helpers.fake_ai_client import FakeAIClient


def test_malformed_json_triggers_audit(monkeypatch, tmp_path):
    audit = create_audit_logger("test")
    monkeypatch.setattr(
        "logic.generate_strategy_report.fix_draft_with_guardrails",
        lambda *a, **k: None,
    )

    fake = FakeAIClient()
    fake.add_chat_response("{bad json")
    gen = StrategyGenerator(ai_client=fake)
    gen.generate({}, {}, audit=audit)
    audit_file = audit.save(tmp_path)
    data = json.loads(audit_file.read_text())
    stages = [s["stage"] for s in data["steps"]]
    assert "strategist_raw_output" in stages
    fail_entry = next(s for s in data["steps"] if s["stage"] == "strategist_failure")
    assert (
        fail_entry["details"].get("failure_reason")
        == StrategistFailureReason.UNRECOGNIZED_FORMAT.value
    )
