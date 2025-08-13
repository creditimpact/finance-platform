import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from backend.core.logic.strategy.normalizer_2_5 import normalize_and_tag


class DummyRulebook:
    def __init__(self, version: str = "2024-01") -> None:
        self.version = version


def test_emits_rule_event(monkeypatch):
    events = []

    def fake_emit(event, payload):
        events.append((event, payload))

    monkeypatch.setattr(
        "backend.core.logic.strategy.normalizer_2_5.emit_event", fake_emit
    )
    normalize_and_tag({}, {}, DummyRulebook(), account_id="123")
    assert events
    assert events[0][1] == {
        "account_id": "123",
        "rule_hits": [],
        "rulebook_version": "2024-01",
    }
