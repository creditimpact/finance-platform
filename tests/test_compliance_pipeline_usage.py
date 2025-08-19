import sys
import types
from pathlib import Path

import pdfkit
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from tests.helpers.fake_ai_client import FakeAIClient


@pytest.mark.parametrize("doc_type", ["dispute", "instructions", "goodwill"])
def test_pipeline_invoked_for_documents(monkeypatch, tmp_path, doc_type):
    calls = []
    monkeypatch.setattr(pdfkit, "configuration", lambda *a, **k: None)
    monkeypatch.setattr(
        "backend.core.logic.strategy.fallback_manager.determine_fallback_action",
        lambda *a, **k: "dispute",
    )
    utils_pkg = types.ModuleType("backend.core.logic.letters.utils")
    pdf_ops_mod = types.SimpleNamespace(
        gather_supporting_docs=lambda *a, **k: ("", [], None)
    )
    utils_pkg.pdf_ops = pdf_ops_mod
    monkeypatch.setitem(sys.modules, "backend.core.logic.letters.utils", utils_pkg)
    monkeypatch.setitem(
        sys.modules, "backend.core.logic.letters.utils.pdf_ops", pdf_ops_mod
    )
    monkeypatch.setattr(
        "backend.core.logic.rendering.letter_rendering.render_dispute_letter_html",
        lambda *a, **k: "",
    )

    if doc_type == "dispute":
        from backend.core.logic.letters.letter_generator import (
            generate_all_dispute_letters_with_ai,
        )

        monkeypatch.setattr(
            "backend.core.logic.letters.letter_generator.run_compliance_pipeline",
            lambda html, state, session_id, dt, ai_client=None: calls.append(dt)
            or html,
        )
        monkeypatch.setattr(
            "backend.core.logic.letters.letter_generator.generate_strategy",
            lambda session_id, bureau_data: {"dispute_items": {"1": {}}},
        )

        def fake_call_gpt(*a, **k):
            return {
                "opening_paragraph": "Opening",
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
                "closing_paragraph": "Closing",
            }

        monkeypatch.setattr(
            "backend.core.logic.letters.letter_generator.call_gpt_dispute_letter",
            fake_call_gpt,
        )
        monkeypatch.setattr(
            "backend.core.logic.rendering.pdf_renderer.render_html_to_pdf",
            lambda html, path: None,
        )
        from backend.core.logic.strategy.summary_classifier import ClassificationRecord
        from backend.core.models import BureauPayload, ClientInfo

        client = ClientInfo.from_dict({"name": "Client", "session_id": "s1"})
        bureau_data = {
            "Experian": BureauPayload.from_dict(
                {
                    "disputes": [
                        {
                            "name": "Bank A",
                            "account_number": "1",
                            "account_id": "1",
                            "action_tag": "dispute",
                        }
                    ],
                    "inquiries": [],
                }
            )
        }
        classification_map = {
            "1": ClassificationRecord({}, {"category": "not_mine"}, "")
        }
        with pytest.warns(UserWarning):
            generate_all_dispute_letters_with_ai(
                client,
                bureau_data,
                tmp_path,
                False,
                None,
                ai_client=FakeAIClient(),
                classification_map=classification_map,
            )
    elif doc_type == "instructions":
        from backend.core.logic.rendering.instructions_generator import (
            generate_instruction_file,
        )

        monkeypatch.setattr(
            "backend.core.logic.rendering.instructions_generator.run_compliance_pipeline",
            lambda html, state, session_id, dt, ai_client=None: calls.append(dt)
            or html,
        )
        monkeypatch.setattr(
            "backend.core.logic.rendering.instruction_data_preparation.generate_account_action",
            lambda acc, ai_client=None: "Pay the balance",
        )
        monkeypatch.setattr(
            "backend.core.logic.rendering.pdf_renderer.render_html_to_pdf",
            lambda html, path: None,
        )
        from backend.core.models import ClientInfo

        client = ClientInfo.from_dict({"name": "Client", "session_id": "s2"})
        bureau_data = {
            "Experian": {
                "all_accounts": [
                    {
                        "name": "Bank B",
                        "status": "good",
                        "action_tag": "positive",
                    }
                ],
                "disputes": [],
                "inquiries": [],
            }
        }
        generate_instruction_file(
            client, bureau_data, False, tmp_path / "inst", ai_client=FakeAIClient()
        )
    else:  # goodwill
        from backend.core.logic.letters.generate_goodwill_letters import (
            generate_goodwill_letter_with_ai,
        )

        monkeypatch.setattr(
            "backend.core.logic.letters.generate_goodwill_letters.run_compliance_pipeline",
            lambda html, state, session_id, dt, ai_client=None: calls.append(dt)
            or html,
        )
        monkeypatch.setattr(
            "backend.core.logic.letters.generate_goodwill_letters.goodwill_prompting.generate_goodwill_letter_draft",
            lambda *a, **k: (
                {
                    "intro_paragraph": "Intro",
                    "hardship_paragraph": "Hard",
                    "recovery_paragraph": "Rec",
                    "closing_paragraph": "Close",
                    "accounts": [],
                },
                [],
            ),
        )
        monkeypatch.setattr(
            "backend.core.logic.rendering.pdf_renderer.render_html_to_pdf",
            lambda html, path: None,
        )
        monkeypatch.setattr(
            "backend.core.logic.letters.generate_goodwill_letters.gather_supporting_docs",
            lambda session_id: ("", [], None),
        )
        output = tmp_path / "gw"
        output.mkdir()
        generate_goodwill_letter_with_ai(
            "Creditor",
            [{"name": "Creditor", "account_number": "1"}],
            {"name": "Client", "session_id": "s3"},
            output,
            ai_client=FakeAIClient(),
            strategy={
                "accounts": [
                    {
                        "name": "Creditor",
                        "account_number": "1",
                        "action_tag": "goodwill",
                    }
                ]
            },
        )

    assert doc_type in calls
