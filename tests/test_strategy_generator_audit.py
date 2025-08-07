import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from audit import start_audit, clear_audit
from logic.generate_strategy_report import StrategyGenerator


def test_malformed_json_triggers_audit(monkeypatch, tmp_path):
    audit = start_audit()
    monkeypatch.setattr(
        "logic.generate_strategy_report.fix_draft_with_guardrails",
        lambda *a, **k: None,
    )

    class DummyResponse:
        class Choice:
            class Message:
                content = "{bad json"
            message = Message()
        choices = [Choice()]

    monkeypatch.setattr(
        "logic.generate_strategy_report.client.chat.completions.create",
        lambda *a, **k: DummyResponse(),
    )

    gen = StrategyGenerator()
    gen.generate({}, {}, audit=audit)
    audit_file = audit.save(tmp_path)
    data = json.loads(audit_file.read_text())
    stages = [s["stage"] for s in data["steps"]]
    assert "strategist_raw_output" in stages
    assert "strategist_failure" in stages
    clear_audit()
