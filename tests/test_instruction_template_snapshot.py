from pathlib import Path

from backend.core.logic.rendering.instruction_renderer import render_instruction_html
from backend.core.letters.router import select_template


def test_instruction_template_snapshot():
    ctx = {
        "client_name": "John Doe",
        "date": "January 1, 2024",
        "accounts_summary": {
            "problematic": [
                {
                    "name": "Bank",
                    "bureaus": ["Experian"],
                    "status": "Open",
                    "late_payments": {},
                    "recommended_action": None,
                    "needs_evidence": [],
                    "letters": [],
                    "dispute_type": "",
                    "utilization": None,
                    "advisor_comment": None,
                    "personal_note": None,
                    "action_sentence": "Pay the balance.",
                }
            ],
            "improve": [],
            "positive": [],
        },
        "per_account_actions": [
            {"account_ref": "Bank", "action_sentence": "Pay the balance."}
        ],
    }
    decision = select_template("instruction", ctx, phase="finalize")
    html = render_instruction_html(ctx, decision.template_path)
    expected = Path("tests/golden_instruction.html").read_text()
    assert html.strip() == expected.strip()
