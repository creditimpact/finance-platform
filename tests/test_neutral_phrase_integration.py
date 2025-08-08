from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from logic.letter_generator import call_gpt_dispute_letter
from logic import rules_loader
from tests.helpers.fake_ai_client import FakeAIClient
from models.account import Account


def test_dispute_prompt_includes_neutral_phrase(monkeypatch):
    fake_client = FakeAIClient()
    fake_client.add_chat_response(
        '{"opening_paragraph": "", "accounts": [], "inquiries": [], "closing_paragraph": ""}'
    )

    monkeypatch.setattr(
        "logic.letter_generator.classify_client_summary",
        lambda struct, state: {"category": "not_mine"},
    )

    structured = {"explanation": "I never opened this"}
    call_gpt_dispute_letter(
        {"legal_name": "Test", "session_id": ""},
        "Equifax",
        [
            Account(
                account_id="1", name="Acc", account_number="123", reported_status="open"
            )
        ],
        [],
        False,
        {"1": structured},
        "",
        ai_client=fake_client,
    )

    prompt = fake_client.chat_payloads[0]["messages"][0]["content"]
    phrases = rules_loader.load_neutral_phrases()["not_mine"]
    assert any(p in prompt for p in phrases)
    assert '"explanation": "I never opened this"' in prompt
