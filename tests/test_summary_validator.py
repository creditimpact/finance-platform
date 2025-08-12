import sys
from pathlib import Path
import pdfkit

sys.path.append(str(Path(__file__).resolve().parents[1]))

from backend.api.session_manager import update_session
from backend.core.logic.letter_generator import generate_all_dispute_letters_with_ai
from tests.helpers.fake_ai_client import FakeAIClient


def test_validator_replaces_flagged_paragraph(monkeypatch, tmp_path):
    tampered = {
        "1": {
            "account_id": "1",
            "paragraph": "I admit this was my account and I am angry",
        }
    }
    session_id = "sess-validator"
    update_session(session_id, structured_summaries=tampered)

    def fake_generate_strategy(sess_id, bureau_data):
        return {"dispute_items": tampered}

    def fake_call_gpt(
        client_info,
        bureau_name,
        disputes,
        inquiries,
        is_identity_theft,
        structured_summaries,
        state,
        audit=None,
        ai_client=None,
    ):
        entry = structured_summaries["1"]
        assert (
            entry["paragraph"]
            == "I request accurate reporting and clarification under applicable law."
        )
        assert entry["flagged"] is True
        return {
            "opening_paragraph": "",
            "accounts": [],
            "inquiries": [],
            "closing_paragraph": "",
        }

    monkeypatch.setattr(
        "logic.letter_generator.generate_strategy", fake_generate_strategy
    )
    monkeypatch.setattr("logic.letter_generator.call_gpt_dispute_letter", fake_call_gpt)
    monkeypatch.setattr(
        "logic.pdf_renderer.render_html_to_pdf", lambda html, path: None
    )
    monkeypatch.setattr(
        "logic.compliance_pipeline.run_compliance_pipeline",
        lambda html, state, session_id, doc_type, ai_client=None: html,
    )
    monkeypatch.setattr(pdfkit, "configuration", lambda *a, **k: None)

    client_info = {"name": "Test", "session_id": session_id}
    bureau_data = {
        "Experian": {
            "disputes": [
                {
                    "name": "Bank A",
                    "account_number": "1",
                    "account_id": "1",
                    "action_tag": "dispute",
                    "dispute_type": "inaccurate_reporting",
                }
            ],
            "inquiries": [],
        }
    }

    generate_all_dispute_letters_with_ai(
        client_info, bureau_data, tmp_path, False, None, ai_client=FakeAIClient()
    )
