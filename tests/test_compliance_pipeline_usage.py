import sys
from pathlib import Path
import types

import pdfkit
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

sys.modules.setdefault(
    "backend.core.logic.letters.fallback_manager",
    types.SimpleNamespace(determine_fallback_action=lambda *a, **k: "dispute"),
)
sys.modules.setdefault(
    "backend.core.logic.letters.summary_classifier",
    types.SimpleNamespace(classify_client_summary=lambda struct, state=None: {"category": "not_mine"}),
)

utils_pkg = types.ModuleType("backend.core.logic.letters.utils")
pdf_ops_mod = types.SimpleNamespace(gather_supporting_docs=lambda *a, **k: ("", [], None))
utils_pkg.pdf_ops = pdf_ops_mod
sys.modules.setdefault("backend.core.logic.letters.utils", utils_pkg)
sys.modules.setdefault("backend.core.logic.letters.utils.pdf_ops", pdf_ops_mod)
sys.modules.setdefault(
    "backend.core.logic.letters.letter_rendering",
    types.SimpleNamespace(render_dispute_letter_html=lambda *a, **k: ""),
)

from tests.helpers.fake_ai_client import FakeAIClient


@pytest.mark.parametrize("doc_type", ["dispute", "instructions", "goodwill"])
def test_pipeline_invoked_for_documents(monkeypatch, tmp_path, doc_type):
    calls = []
    monkeypatch.setattr(pdfkit, "configuration", lambda *a, **k: None)

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
        monkeypatch.setattr(
            "backend.core.logic.letters.letter_generator.call_gpt_dispute_letter",
            lambda *a, **k: {
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
            },
        )
        monkeypatch.setattr(
            "backend.core.logic.rendering.pdf_renderer.render_html_to_pdf",
            lambda html, path: None,
        )
        from backend.core.models import ClientInfo, BureauPayload

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
        with pytest.warns(UserWarning):
            generate_all_dispute_letters_with_ai(
                client, bureau_data, tmp_path, False, None, ai_client=FakeAIClient()
            )
    elif doc_type == "instructions":
        from backend.core.logic.rendering.instructions_generator import generate_instruction_file

        monkeypatch.setattr(
            "backend.core.logic.rendering.instructions_generator.run_compliance_pipeline",
            lambda html, state, session_id, dt, ai_client=None: calls.append(dt)
            or html,
        )
        monkeypatch.setattr(
            "backend.core.logic.rendering.instruction_data_preparation.generate_account_action",
            lambda acc, ai_client=None: "Do something",
        )
        monkeypatch.setattr(
            "backend.core.logic.rendering.pdf_renderer.render_html_to_pdf",
            lambda html, path: None,
        )
        from backend.core.models import ClientInfo, BureauPayload

        client = ClientInfo.from_dict({"name": "Client", "session_id": "s2"})
        bureau_data = {
            "Experian": BureauPayload.from_dict(
                {
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
            )
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
        )

    assert doc_type in calls
