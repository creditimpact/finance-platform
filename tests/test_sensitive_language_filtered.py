import json
import sys
from pathlib import Path

import pdfkit

sys.path.append(str(Path(__file__).resolve().parents[1]))

from session_manager import update_session, update_intake
from logic.letter_generator import generate_all_dispute_letters_with_ai
from logic.generate_goodwill_letters import generate_goodwill_letter_with_ai


def test_dispute_letter_ignores_emotional_text(monkeypatch, tmp_path):
    structured = {
        "1": {
            "account_id": "1",
            "dispute_type": "late",
            "facts_summary": "summary",
            "claimed_errors": [],
            "dates": {},
            "evidence": [],
            "risk_flags": {},
        }
    }
    session_id = "sess-emotion-dispute"
    update_session(session_id, structured_summaries=structured)
    update_intake(
        session_id,
        raw_explanations=[{"account_id": "1", "text": "I am furious and heartbroken"}],
    )

    def fake_generate_strategy(sess_id, bureau_data):
        return {"dispute_items": structured}

    def fake_call_gpt(client_info, bureau_name, disputes, inquiries, is_identity_theft, structured_summaries, state):
        text = json.dumps(client_info)
        assert "furious" not in text
        assert "heartbroken" not in text
        return {
            "opening_paragraph": "Opening",
            "accounts": [
                {
                    "name": "Bank A",
                    "account_number": "1",
                    "status": "open",
                    "paragraph": "standard",
                    "requested_action": "Delete",
                }
            ],
            "inquiries": [],
            "closing_paragraph": "Closing",
        }

    monkeypatch.setattr("logic.letter_generator.generate_strategy", fake_generate_strategy)
    monkeypatch.setattr("logic.letter_generator.call_gpt_dispute_letter", fake_call_gpt)
    monkeypatch.setattr("logic.letter_generator.render_html_to_pdf", lambda html, path: None)
    monkeypatch.setattr("logic.letter_generator.fix_draft_with_guardrails", lambda *a, **k: None)
    monkeypatch.setattr(pdfkit, "configuration", lambda *a, **k: None)

    client_info = {
        "name": "Test Client",
        "session_id": session_id,
        "custom_dispute_notes": {"Bank A": "I am furious and heartbroken"},
    }
    bureau_data = {
        "Experian": {
            "disputes": [
                {"name": "Bank A", "account_number": "1", "action_tag": "dispute"}
            ],
            "inquiries": [],
        }
    }

    generate_all_dispute_letters_with_ai(client_info, bureau_data, tmp_path, False)
    data = json.load(open(tmp_path / "Experian_gpt_response.json"))
    dump = json.dumps(data)
    assert "furious" not in dump
    assert "heartbroken" not in dump


def test_goodwill_letter_ignores_emotional_text(monkeypatch, tmp_path):
    session_id = "sess-emotion-goodwill"
    update_session(session_id, structured_summaries={"1": {"account_id": "1"}})
    update_intake(
        session_id,
        raw_explanations=[{"account_id": "1", "text": "They ruined my life and I'm devastated"}],
    )

    def fake_call_gpt(
        client_name,
        creditor,
        accounts,
        personal_story=None,
        tone="neutral",
        session_id=None,
        structured_summaries=None,
    ):
        assert "devastated" not in (personal_story or "")
        return {
            "intro_paragraph": "Intro",
            "hardship_paragraph": "Hardship",
            "recovery_paragraph": "Recovery",
            "closing_paragraph": "Closing",
            "accounts": [],
        }

    monkeypatch.setattr(
        "logic.generate_goodwill_letters.call_gpt_for_goodwill_letter", fake_call_gpt
    )
    monkeypatch.setattr(
        "logic.generate_goodwill_letters.render_html_to_pdf", lambda html, path: None
    )
    monkeypatch.setattr(
        "logic.generate_goodwill_letters.fix_draft_with_guardrails", lambda *a, **k: None
    )
    monkeypatch.setattr(pdfkit, "configuration", lambda *a, **k: None)

    client_info = {
        "name": "Test Client",
        "session_id": session_id,
        "custom_dispute_notes": {"Creditor": "I am devastated and angry"},
    }
    accounts = [
        {"name": "Creditor", "account_number": "1", "action_tag": "goodwill"}
    ]

    generate_goodwill_letter_with_ai("Creditor", accounts, client_info, tmp_path)
    data = json.load(open(tmp_path / "Creditor_gpt_response.json"))
    dump = json.dumps(data)
    assert "devastated" not in dump
    assert "angry" not in dump

