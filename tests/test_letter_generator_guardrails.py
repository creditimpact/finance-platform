import json
from pathlib import Path

import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from logic.guardrails import generate_letter_with_guardrails, fix_draft_with_guardrails
from session_manager import update_session, get_session
from tests.helpers.fake_ai_client import FakeAIClient


def test_autofix_and_no_raw_explanation(monkeypatch, tmp_path):
    session_id = "sess-test"
    structured = {
        "account_id": "1",
        "dispute_type": "not_mine",
        "facts_summary": "billing issue",
    }
    update_session(session_id, structured_summaries={"1": structured})
    fake = FakeAIClient()
    fake.add_chat_response("I admit fault. SSN 123-45-6789.")
    fake.add_chat_response("This is a clean letter.")

    prompt = "Here is the structured summary for the account:\n" + json.dumps(
        structured
    )

    text, violations, iters = generate_letter_with_guardrails(
        prompt, "CA", {"debt_type": "medical"}, session_id, "dispute", ai_client=fake
    )

    assert iters == 2
    assert all(v["severity"] != "critical" for v in violations)
    session = get_session(session_id)
    stored = session["letters_generated"][0]["text"]
    assert "I admit" not in stored
    assert "123-45-6789" not in stored


def test_state_clause_added(monkeypatch):
    session_id = "sess-ny"
    update_session(session_id, structured_summaries={})

    fake2 = FakeAIClient()
    fake2.add_chat_response("Irrelevant")

    text, violations, _ = fix_draft_with_guardrails(
        "Please investigate.",
        "NY",
        {"debt_type": "medical"},
        session_id,
        "dispute",
        ai_client=fake2,
    )
    assert "new york financial services law" in text.lower()
