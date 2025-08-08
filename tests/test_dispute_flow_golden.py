from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from models.account import Account, Inquiry
from logic.letter_rendering import render_dispute_letter_html


class FakeAI:
    class Choice:
        class Message:
            def __init__(self, content):
                self.content = content

        def __init__(self, content):
            self.message = FakeAI.Choice.Message(content)

    def __init__(self, payload):
        self.payload = payload

    def chat_completion(self, **kwargs):
        return type("Resp", (), {"choices": [FakeAI.Choice(self.payload)]})


def test_dispute_flow_golden(monkeypatch):
    import types
    import sys

    sys.modules["fitz"] = types.ModuleType("fitz")
    sys.modules["pymupdf"] = types.ModuleType("pymupdf")
    from logic.gpt_prompting import call_gpt_dispute_letter
    from logic.compliance_pipeline import run_compliance_pipeline

    payload = """{\n  \"opening_paragraph\": \"I dispute the following accounts\",\n  \"accounts\": [{\n    \"name\": \"ABC Bank\",\n    \"account_number\": \"1234\",\n    \"status\": \"Late\",\n    \"paragraph\": \"Please delete\",\n    \"requested_action\": \"Delete\"\n  }],\n  \"inquiries\": [{\n    \"creditor_name\": \"XYZ Bank\",\n    \"date\": \"2020-01-01\"\n  }],\n  \"closing_paragraph\": \"Thank you\"\n}"""
    fake_ai = FakeAI(payload)
    ctx = call_gpt_dispute_letter(
        {},
        "Experian",
        [
            Account(
                name="ABC Bank",
                account_id="1",
                account_number="1234",
                reported_status="Late",
            )
        ],
        [Inquiry(creditor_name="XYZ Bank", date="2020-01-01")],
        False,
        {},
        "CA",
        ai_client=fake_ai,
    )
    ctx.client_name = "John Doe"
    ctx.client_address_lines = ["123 Main St", "Town, ST 12345"]
    ctx.bureau_name = "Experian"
    ctx.bureau_address = "Address"
    ctx.date = "January 1, 2024"
    artifact = render_dispute_letter_html(ctx)
    monkeypatch.setattr(
        "logic.compliance_pipeline.fix_draft_with_guardrails",
        lambda *a, **k: None,
    )
    run_compliance_pipeline(artifact, "CA", "sess", "dispute")
    html = artifact.html
    expected = Path("tests/golden_letter.html").read_text()
    assert html == expected
