import pytest

from backend.core.logic.letters.generate_custom_letters import generate_custom_letter
from backend.core.logic.letters.generate_goodwill_letters import (
    generate_goodwill_letters,
)
from backend.core.logic.letters.letter_generator import (
    generate_all_dispute_letters_with_ai,
)
from backend.core.logic.letters.utils import StrategyContextMissing
from tests.helpers.fake_ai_client import FakeAIClient


@pytest.fixture(autouse=True)
def _set_enforcement(monkeypatch):
    monkeypatch.setenv("STAGE4_POLICY_ENFORCEMENT", "1")
    monkeypatch.setattr("backend.api.config.STAGE4_POLICY_ENFORCEMENT", True)
    monkeypatch.setattr(
        "backend.core.logic.letters.letter_generator.api_config.STAGE4_POLICY_ENFORCEMENT",
        True,
    )
    monkeypatch.setattr(
        "backend.core.logic.letters.generate_goodwill_letters.api_config.STAGE4_POLICY_ENFORCEMENT",
        True,
    )
    monkeypatch.setattr(
        "backend.core.logic.letters.generate_custom_letters.api_config.STAGE4_POLICY_ENFORCEMENT",
        True,
    )
    events: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        "backend.audit.audit.emit_event", lambda e, p: events.append((e, p))
    )
    monkeypatch.setattr(
        "backend.core.logic.letters.letter_generator.emit_event",
        lambda e, p: events.append((e, p)),
    )
    monkeypatch.setattr(
        "backend.core.logic.letters.generate_goodwill_letters.emit_event",
        lambda e, p: events.append((e, p)),
    )
    monkeypatch.setattr(
        "backend.core.logic.letters.generate_custom_letters.emit_event",
        lambda e, p: events.append((e, p)),
    )
    return events


def test_dispute_letter_requires_strategy(tmp_path, monkeypatch, _set_enforcement):
    events = _set_enforcement
    monkeypatch.setattr(
        "backend.core.logic.letters.letter_generator.generate_strategy",
        lambda session_id, bureau_data: {
            "accounts": [{"account_id": "1"}],
            "dispute_items": {},
        },
    )
    fake = FakeAIClient()
    client = {"name": "Test", "session_id": "s1"}
    bureau_map = {"Experian": {"disputes": [], "inquiries": []}}
    with pytest.raises(StrategyContextMissing):
        generate_all_dispute_letters_with_ai(
            client, bureau_map, tmp_path, False, None, ai_client=fake
        )
    assert events == [
        ("strategy_context_missing", {"account_id": "1", "letter_type": "dispute"})
    ]
    assert not any(tmp_path.iterdir())


def test_goodwill_letter_requires_strategy(tmp_path, _set_enforcement):
    events = _set_enforcement
    client = {"legal_name": "John Doe", "session_id": "s1"}
    bureau_map = {
        "Experian": {
            "disputes": [{"name": "Bank", "account_number": "1", "account_id": "1"}]
        }
    }
    fake = FakeAIClient()
    with pytest.raises(StrategyContextMissing):
        generate_goodwill_letters(
            client, bureau_map, tmp_path, audit=None, ai_client=fake
        )
    assert events[-1] == (
        "strategy_context_missing",
        {"account_id": "1", "letter_type": "goodwill"},
    )
    assert not any(tmp_path.iterdir())


def test_custom_letter_requires_strategy(tmp_path, _set_enforcement):
    events = _set_enforcement
    account = {"name": "Bank", "account_number": "1", "account_id": "1"}
    client = {"name": "Tester", "session_id": "s1"}
    fake = FakeAIClient()
    with pytest.raises(StrategyContextMissing):
        generate_custom_letter(account, client, tmp_path, audit=None, ai_client=fake)
    assert events[-1] == (
        "strategy_context_missing",
        {"account_id": "1", "letter_type": "custom"},
    )
    assert not any(tmp_path.iterdir())
