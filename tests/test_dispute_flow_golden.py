import sys
import re
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from backend.core.logic.rendering.letter_rendering import render_dispute_letter_html
from backend.core.letters.router import select_template
from backend.core.models.account import Account, Inquiry


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
    import sys
    import types

    sys.modules["fitz"] = types.ModuleType("fitz")
    sys.modules["pymupdf"] = types.ModuleType("pymupdf")
    from backend.core.logic.compliance.compliance_pipeline import (
        run_compliance_pipeline,
    )
    from backend.core.logic.letters.gpt_prompting import call_gpt_dispute_letter

    payload = (
        """{\n  \"opening_paragraph\": \"Under FCRA §611, I dispute the following accounts and request an investigation. Please respond within 30 days.\",\n  \"accounts\": [{\n    \"name\": \"ABC Bank\",\n    \"account_number\": \"1234\",\n    \"status\": \"Late\",\n    \"paragraph\": \"Please delete\",\n    \"requested_action\": \"Delete\"\n  }],\n  \"inquiries\": [{\n    \"creditor_name\": \"XYZ Bank\",\n    \"date\": \"2020-01-01\"\n  }],\n  \"closing_paragraph\": \"Thank you\"\n}"""
    )
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
        {},
        ai_client=fake_ai,
    )
    ctx.client_name = "John Doe"
    ctx.client_address_lines = ["123 Main St", "Town, ST 12345"]
    ctx.bureau_name = "Experian"
    ctx.bureau_address = "Address"
    ctx.date = "January 1, 2024"
    decision = select_template(
        "dispute", {"bureau": "Experian"}, phase="finalize"
    )
    ctx_dict = ctx.to_dict()
    ctx_dict["account_number_masked"] = "1234"
    artifact = render_dispute_letter_html(
        ctx_dict, decision.template_path
    )
    monkeypatch.setattr(
        "backend.core.logic.compliance.compliance_pipeline.fix_draft_with_guardrails",
        lambda *a, **k: None,
    )
    from tests.helpers.fake_ai_client import FakeAIClient

    run_compliance_pipeline(artifact, "CA", "sess", "dispute", ai_client=FakeAIClient())
    html = re.sub(r"\s+", " ", artifact.html).strip()
    expected = re.sub(
        r"\s+", " ", Path("tests/golden_letter.html").read_text()
    ).strip()
    assert html == expected
