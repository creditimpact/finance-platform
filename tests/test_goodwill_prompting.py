import json
from backend.core.logic.letters.goodwill_prompting import generate_goodwill_letter_draft
from tests.helpers.fake_ai_client import FakeAIClient


def test_prompting_parses_ai_response(monkeypatch):
    ai = FakeAIClient()
    ai.add_chat_response(
        json.dumps(
            {"intro_paragraph": "hello", "accounts": [], "closing_paragraph": "bye"}
        )
    )

    def fake_docs(session_id):
        return "", [], {}

    monkeypatch.setattr(
        "backend.core.logic.letters.goodwill_prompting.gather_supporting_docs",
        fake_docs,
    )

    summaries = [{"name": "Bank", "account_number": "1", "status": "Open"}]
    result, doc_names = generate_goodwill_letter_draft(
        client_name="John",
        creditor="Bank",
        account_summaries=summaries,
        session_id="s1",
        ai_client=ai,
    )
    assert result["intro_paragraph"] == "hello"
    assert doc_names == []
    assert ai.chat_payloads
