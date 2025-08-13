import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from backend.core.logic.compliance import rules_loader
from backend.core.logic.letters.letter_generator import call_gpt_dispute_letter
from backend.core.logic.strategy.summary_classifier import ClassificationRecord
from backend.core.models.account import Account
from tests.helpers.fake_ai_client import FakeAIClient


def test_dispute_prompt_includes_neutral_phrase(monkeypatch):
    fake_client = FakeAIClient()
    fake_client.add_chat_response(
        '{"opening_paragraph": "", "accounts": [], "inquiries": [], "closing_paragraph": ""}'
    )

    structured = {"explanation": "I never opened this"}
    classification_map = {
        "1": ClassificationRecord(structured, {"category": "not_mine"}, "")
    }
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
        classification_map=classification_map,
        ai_client=fake_client,
    )

    prompt = fake_client.chat_payloads[0]["messages"][0]["content"]
    phrases = rules_loader.load_neutral_phrases()["not_mine"]
    assert any(p in prompt for p in phrases)
    assert '"explanation": "I never opened this"' in prompt
