from pathlib import Path
import json
import sys
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from logic.letter_generator import (
    generate_all_dispute_letters_with_ai,
    DEFAULT_DISPUTE_REASON,
)
from tests.helpers.fake_ai_client import FakeAIClient


class Dummy:
    def __init__(self, data):
        self.data = data


def test_unrecognized_action_fallback(monkeypatch, tmp_path, capsys):
    # Patch external dependencies
    monkeypatch.setattr(
        "logic.letter_generator.generate_strategy",
        lambda session_id, bureau_data: {"dispute_items": {"1": {}}},
    )

    def fake_call_gpt(*args, **kwargs):
        return {
            "opening_paragraph": "Opening",
            "accounts": [
                {
                    "name": "Bank A",
                    "account_number": "1",
                    "status": "open",
                    "paragraph": "client raw note ABCXYZ",
                    "requested_action": "Delete",
                }
            ],
            "inquiries": [],
            "closing_paragraph": "Closing",
        }

    monkeypatch.setattr("logic.letter_generator.call_gpt_dispute_letter", fake_call_gpt)
    monkeypatch.setattr(
        "logic.pdf_renderer.render_html_to_pdf", lambda html, path: None
    )
    monkeypatch.setattr(
        "logic.compliance_pipeline.run_compliance_pipeline",
        lambda html, state, session_id, doc_type, ai_client=None: html,
    )
    import pdfkit

    monkeypatch.setattr(pdfkit, "configuration", lambda *a, **k: None)

    client_info = {
        "name": "Test Client",
        "session_id": "sess1",
    }

    bureau_data = {
        "Experian": {
            "disputes": [
                {
                    "name": "Bank A",
                    "account_number": "1",
                    "account_id": "1",
                    "action_tag": "dispute",
                    "fallback_unrecognized_action": True,
                }
            ],
            "inquiries": [],
        }
    }

    with pytest.warns(UserWarning):
        fake = FakeAIClient()
        generate_all_dispute_letters_with_ai(
            client_info, bureau_data, tmp_path, False, None, ai_client=fake
        )

    with open(tmp_path / "Experian_gpt_response.json") as f:
        data = json.load(f)
    assert data["accounts"][0]["paragraph"] == DEFAULT_DISPUTE_REASON

    out, _ = capsys.readouterr()
    assert "fallback_used=True" in out
