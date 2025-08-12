from backend.core.logic.goodwill_rendering import render_goodwill_letter
from tests.helpers.fake_ai_client import FakeAIClient


def test_rendering_calls_compliance_and_pdf(tmp_path):
    html_called = {}

    def fake_pdf(html, path):
        html_called["pdf"] = (html, path)

    def fake_compliance(html, state, session_id, doc_type, ai_client=None):
        html_called["compliance"] = html
        return html

    gpt_data = {
        "intro_paragraph": "hi",
        "hardship_paragraph": "hard",
        "recovery_paragraph": "rec",
        "closing_paragraph": "bye",
        "accounts": [
            {"name": "Bank", "account_number": "1", "status": "Open", "paragraph": "p"}
        ],
    }
    client_info = {"legal_name": "John Doe", "session_id": "s1", "state": "CA"}
    render_goodwill_letter(
        "Bank",
        gpt_data,
        client_info,
        tmp_path,
        doc_names=["Doc1.pdf"],
        ai_client=FakeAIClient(),
        compliance_fn=fake_compliance,
        pdf_fn=fake_pdf,
    )
    assert "compliance" in html_called
    assert "Doc1.pdf" in html_called["compliance"]
    json_path = tmp_path / "Bank_gpt_response.json"
    assert json_path.exists()

    gpt_data = {
        "intro_paragraph": "hi",
        "hardship_paragraph": "hard",
        "recovery_paragraph": "rec",
        "closing_paragraph": "bye",
        "accounts": [
            {"name": "Bank", "account_number": "1", "status": "Open", "paragraph": "p"}
        ],
    }
    client_info = {"legal_name": "John Doe", "session_id": "s1", "state": "CA"}
    render_goodwill_letter(
        "Bank",
        gpt_data,
        client_info,
        tmp_path,
        doc_names=["Doc1.pdf"],
        ai_client=FakeAIClient(),
    )
    assert "compliance" in html_called
    assert "Doc1.pdf" in html_called["compliance"]
    json_path = tmp_path / "Bank_gpt_response.json"
    assert json_path.exists()
