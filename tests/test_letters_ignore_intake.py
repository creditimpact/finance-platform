import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from session_manager import update_session, update_intake
from logic.letter_generator import generate_all_dispute_letters_with_ai
from tests.helpers.fake_ai_client import FakeAIClient


def test_letters_do_not_access_raw_intake(monkeypatch, tmp_path):
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
    session_id = "sess-intake-guard"
    update_session(session_id, structured_summaries=structured)
    update_intake(session_id, raw_explanations=[{"account_id": "1", "text": "SECRET RAW"}])

    def fake_generate_strategy(sess_id, bureau_data):
        return {"dispute_items": structured}

    def fake_call_gpt(client_info, bureau_name, disputes, inquiries, is_identity_theft, structured_summaries, state, ai_client=None):
        text = json.dumps(structured_summaries)
        assert "SECRET RAW" not in text
        return {
            "opening_paragraph": "Opening",
            "accounts": [
                {
                    "name": "Bank A",
                    "account_number": "1",
                    "status": "open",
                    "paragraph": "placeholder",
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
    import pdfkit
    monkeypatch.setattr(pdfkit, "configuration", lambda *a, **k: None)

    client_info = {"name": "Test Client", "session_id": session_id}
    bureau_data = {
        "Experian": {
            "disputes": [
                {"name": "Bank A", "account_number": "1", "account_id": "1", "action_tag": "dispute"}
            ],
            "inquiries": [],
        }
    }

    fake = FakeAIClient()
    generate_all_dispute_letters_with_ai(client_info, bureau_data, tmp_path, False, ai_client=fake)
    with open(tmp_path / "Experian_gpt_response.json") as f:
        data = json.load(f)
    assert "SECRET RAW" not in json.dumps(data)
