import pytest

from backend.core.logic.letters.generate_custom_letters import generate_custom_letter
from backend.core.logic.letters.generate_goodwill_letters import (
    generate_goodwill_letters,
)
from backend.core.logic.letters.letter_generator import (
    generate_all_dispute_letters_with_ai,
)
from backend.core.logic.letters.exceptions import StrategyContextMissing
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


def test_dispute_requires_action_tag(tmp_path, monkeypatch):
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


def test_goodwill_requires_action_tag(tmp_path):
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


def test_custom_requires_action_tag(tmp_path):
    account = {"name": "Bank", "account_number": "1", "account_id": "1"}
    client = {"name": "Tester", "session_id": "s1"}
    fake = FakeAIClient()
    with pytest.raises(StrategyContextMissing):
        generate_custom_letter(account, client, tmp_path, audit=None, ai_client=fake)
