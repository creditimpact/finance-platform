import pdfkit

from session_manager import update_session, update_intake
from logic.letter_generator import generate_all_dispute_letters_with_ai
from tests.helpers.fake_ai_client import FakeAIClient


def _setup(monkeypatch):
    monkeypatch.setattr(
        "logic.pdf_renderer.render_html_to_pdf", lambda html, path: None
    )
    monkeypatch.setattr(
        "logic.compliance_pipeline.run_compliance_pipeline",
        lambda html, state, session_id, doc_type, ai_client=None: html,
    )
    monkeypatch.setattr(pdfkit, "configuration", lambda *a, **k: None)


def test_warning_on_raw_client_text(monkeypatch, tmp_path, recwarn):
    structured = {"1": {"account_id": "1"}}
    session_id = "sess-warn-raw"
    update_session(session_id, structured_summaries=structured)
    update_intake(
        session_id,
        raw_explanations=[{"account_id": "1", "text": "very sensitive info"}],
    )

    def fake_strategy(sess_id, bureau_data):
        return {"dispute_items": structured}

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
        return {
            "opening_paragraph": "open",
            "accounts": [
                {
                    "name": "Bank A",
                    "account_number": "1",
                    "status": "open",
                    "paragraph": "p",
                    "requested_action": "Delete",
                }
            ],
            "inquiries": [],
            "closing_paragraph": "close",
        }

    _setup(monkeypatch)
    monkeypatch.setattr("logic.letter_generator.generate_strategy", fake_strategy)
    monkeypatch.setattr("logic.letter_generator.call_gpt_dispute_letter", fake_call_gpt)

    client_info = {
        "name": "Client",
        "session_id": session_id,
        "custom_dispute_notes": {"Bank A": "very sensitive info"},
    }
    bureau_data = {
        "Experian": {
            "disputes": [
                {"name": "Bank A", "account_number": "1", "action_tag": "dispute"}
            ],
            "inquiries": [],
        }
    }

    fake = FakeAIClient()
    generate_all_dispute_letters_with_ai(
        client_info, bureau_data, tmp_path, False, None, ai_client=fake
    )
    assert any("[PolicyViolation]" in str(w.message) for w in recwarn)


def test_warning_on_missing_summary(monkeypatch, tmp_path, recwarn):
    structured = {"1": "not a dict"}
    session_id = "sess-missing-summary"
    update_session(session_id, structured_summaries=structured)

    def fake_strategy(sess_id, bureau_data):
        return {"dispute_items": structured}

    captured = {}

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
        captured["disputes"] = disputes
        return {
            "opening_paragraph": "o",
            "accounts": [],
            "inquiries": [],
            "closing_paragraph": "c",
        }

    _setup(monkeypatch)
    monkeypatch.setattr("logic.letter_generator.generate_strategy", fake_strategy)
    monkeypatch.setattr("logic.letter_generator.call_gpt_dispute_letter", fake_call_gpt)

    client_info = {"name": "Client", "session_id": session_id}
    bureau_data = {
        "Experian": {
            "disputes": [
                {"name": "Bank A", "account_number": "1", "action_tag": "dispute"}
            ],
            "inquiries": [],
        }
    }

    fake = FakeAIClient()
    generate_all_dispute_letters_with_ai(
        client_info, bureau_data, tmp_path, False, None, ai_client=fake
    )
    assert any("[Sanitization]" in str(w.message) for w in recwarn)


def test_unrecognized_dispute_type_fallback(monkeypatch, tmp_path, recwarn):
    structured = {"1": {"account_id": "1"}}
    session_id = "sess-bad-type"
    update_session(session_id, structured_summaries=structured)

    def fake_strategy(sess_id, bureau_data):
        return {"dispute_items": structured}

    captured = {}

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
        captured["disputes"] = disputes
        return {
            "opening_paragraph": "o",
            "accounts": [],
            "inquiries": [],
            "closing_paragraph": "c",
        }

    _setup(monkeypatch)
    monkeypatch.setattr("logic.letter_generator.generate_strategy", fake_strategy)
    monkeypatch.setattr("logic.letter_generator.call_gpt_dispute_letter", fake_call_gpt)

    client_info = {"name": "Client", "session_id": session_id}
    bureau_data = {
        "Experian": {
            "disputes": [
                {
                    "name": "Bank A",
                    "account_number": "1",
                    "action_tag": "dispute",
                    "dispute_type": "strange",
                }
            ],
            "inquiries": [],
        }
    }

    fake = FakeAIClient()
    generate_all_dispute_letters_with_ai(
        client_info, bureau_data, tmp_path, False, None, ai_client=fake
    )
    assert captured["disputes"][0].dispute_type == "inaccurate_reporting"
    assert any("Unrecognized dispute type" in str(w.message) for w in recwarn)
