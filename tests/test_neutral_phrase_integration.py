import json
from pathlib import Path
import sys
import types

sys.path.append(str(Path(__file__).resolve().parents[1]))

from logic.letter_generator import call_gpt_dispute_letter
from logic import rules_loader
import openai


class DummyResp:
    def __init__(self, content: str):
        self.choices = [types.SimpleNamespace(message=types.SimpleNamespace(content=content))]


def test_dispute_prompt_includes_neutral_phrase(monkeypatch):
    captured = {}

    def fake_create(*args, **kwargs):
        captured["prompt"] = kwargs["messages"][0]["content"]
        return DummyResp('{"opening_paragraph": "", "accounts": [], "inquiries": [], "closing_paragraph": ""}')

    dummy_client = types.SimpleNamespace(
        chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=fake_create))
    )
    monkeypatch.setattr(openai, "OpenAI", lambda **_: dummy_client)
    monkeypatch.setattr(
        "logic.letter_generator.classify_client_summary",
        lambda struct, state: {"category": "not_mine"},
    )

    structured = {"explanation": "I never opened this"}
    call_gpt_dispute_letter(
        {"legal_name": "Test", "session_id": ""},
        "Equifax",
        [{"account_id": "1", "name": "Acc", "account_number": "123", "reported_status": "open"}],
        [],
        False,
        {"1": structured},
        "",
    )

    phrases = rules_loader.load_neutral_phrases()["not_mine"]
    assert any(p in captured["prompt"] for p in phrases)
    assert '"explanation": "I never opened this"' in captured["prompt"]
