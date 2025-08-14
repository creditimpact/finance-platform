import pdfkit
import pytest

from backend.core.logic.letters.letter_generator import (
    generate_dispute_letters_for_all_bureaus,
)
from backend.core.logic.strategy.generate_strategy_report import StrategyGenerator
from backend.core.logic.strategy.summary_classifier import ClassificationRecord
from tests.helpers.fake_ai_client import FakeAIClient


def test_letters_generate_when_strategy_llm_returns_junk(tmp_path, monkeypatch):
    fake_strategy_ai = FakeAIClient()
    fake_strategy_ai.add_chat_response("junk")
    fake_strategy_ai.add_chat_response("still junk")
    gen = StrategyGenerator(ai_client=fake_strategy_ai)

    stage_2_5 = {
        "1": {
            "legal_safe_summary": "",
            "suggested_dispute_frame": "",
            "rule_hits": [],
            "needs_evidence": ["identity_theft_affidavit"],
            "red_flags": [],
            "prohibited_admission_detected": False,
            "rulebook_version": "",
        }
    }

    classification_map = {
        "1": ClassificationRecord(
            {},
            {
                "action_tag": "dispute",
                "priority": "High",
                "legal_notes": ["FCRA 611"],
                "flags": [],
            },
            "",
        )
    }

    client_info = {"name": "Test", "session_id": "sess"}
    bureau_data = {
        "Experian": {
            "disputes": [
                {
                    "name": "Bank",
                    "account_number": "1",
                    "account_id": "1",
                    "status": "open",
                }
            ],
            "inquiries": [],
            "goodwill": [],
            "high_utilization": [],
            "all_accounts": [
                {
                    "name": "Bank",
                    "account_number": "1",
                    "bureaus": ["Experian"],
                    "account_id": "1",
                }
            ],
        }
    }

    monkeypatch.setattr(
        "backend.core.logic.strategy.generate_strategy_report.fix_draft_with_guardrails",
        lambda *a, **k: None,
    )

    strategy = gen.generate(
        client_info,
        bureau_data,
        classification_map=classification_map,
        stage_2_5_data=stage_2_5,
    )

    acc = strategy["accounts"][0]
    assert acc["action_tag"] == "dispute"
    assert acc["priority"] == "High"
    assert acc["legal_notes"] == ["FCRA 611"]
    assert acc["needs_evidence"] == ["identity_theft_affidavit"]

    bureau_data["Experian"]["disputes"][0]["action_tag"] = acc["action_tag"]
    bureau_data["Experian"]["disputes"][0]["recommended_action"] = "Dispute"

    monkeypatch.setattr(
        "backend.core.logic.letters.letter_generator.call_gpt_dispute_letter",
        lambda *a, **k: {
            "opening_paragraph": "",
            "accounts": [],
            "inquiries": [],
            "closing_paragraph": "",
        },
    )
    monkeypatch.setattr(
        "backend.core.logic.rendering.pdf_renderer.render_html_to_pdf",
        lambda html, path: None,
    )
    monkeypatch.setattr(
        "backend.core.logic.compliance.compliance_pipeline.run_compliance_pipeline",
        lambda html, state, session_id, doc_type, ai_client=None: html,
    )
    monkeypatch.setattr(
        "backend.core.logic.letters.letter_generator.generate_strategy",
        lambda session_id, bureau: {"dispute_items": {"1": {}}},
    )
    monkeypatch.setattr(
        "backend.core.logic.letters.letter_generator.render_dispute_letter_html",
        lambda context: "html",
    )

    monkeypatch.setattr(pdfkit, "configuration", lambda *a, **k: None)

    fake_letter_ai = FakeAIClient()
    with pytest.warns(UserWarning):
        generate_dispute_letters_for_all_bureaus(
            client_info,
            bureau_data,
            tmp_path,
            False,
            None,
            ai_client=fake_letter_ai,
            classification_map=classification_map,
        )

    assert (tmp_path / "Experian_gpt_response.json").exists()
