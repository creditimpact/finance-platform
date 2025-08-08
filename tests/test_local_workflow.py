"""Integration-style tests for local workflow."""

# ruff: noqa: E402
import sys
import types
import os
from pathlib import Path
from unittest import mock

sys.path.append(str(Path(__file__).resolve().parents[1]))

# Optional stubs for environments without openai/pdfkit
sys.modules.setdefault("openai", types.SimpleNamespace(OpenAI=lambda *_, **__: None))
sys.modules.setdefault(
    "pdfkit",
    types.SimpleNamespace(
        configuration=lambda *_, **__: None, from_string=lambda *_, **__: None
    ),
)
sys.modules.setdefault(
    "dotenv", types.SimpleNamespace(load_dotenv=lambda *_, **__: None)
)


class _DummyEnv:
    def __init__(self, *_, **__):
        pass

    def get_template(self, *_args, **_kw):
        class _T:
            def render(self, **_):
                return ""

        return _T()


sys.modules.setdefault(
    "jinja2",
    types.SimpleNamespace(
        Environment=_DummyEnv, FileSystemLoader=lambda *_, **__: None
    ),
)
sys.modules.setdefault("fpdf", types.SimpleNamespace(FPDF=object))
sys.modules.setdefault("pdfplumber", types.SimpleNamespace(open=lambda *_, **__: None))
sys.modules.setdefault("fitz", types.SimpleNamespace(open=lambda *_, **__: None))

from logic.letter_generator import (
    generate_dispute_letters_for_all_bureaus,
)  # noqa: E402
from logic.generate_goodwill_letters import generate_goodwill_letters  # noqa: E402
from logic.instructions_generator import generate_instruction_file  # noqa: E402
import logic.instructions_generator as instructions_generator  # noqa: E402
from tests.helpers.fake_ai_client import FakeAIClient  # noqa: E402


def test_minimal_workflow():
    client_info = {
        "name": "Jane Test",
        "email": "jane@example.com",
        "session_id": "test",
    }

    account_dispute = {
        "name": "Bank A",
        "account_number": "1234",
        "status": "Chargeoff",
        "bureaus": ["Experian"],
        "action_tag": "dispute",
        "recommended_action": "Dispute",
        "opened_date": "01/01/2020",
        "account_id": "1",
    }
    account_goodwill = {
        "name": "Card B",
        "account_number": "5678",
        "status": "Closed",
        "bureaus": ["Experian"],
        "action_tag": "goodwill",
        "recommended_action": "Goodwill",
        "late_payments": {"Experian": {"30": 1}},
    }
    # Duplicate of account_dispute under positive to test deduplication
    account_positive = account_dispute.copy()
    account_positive.pop("recommended_action")
    account_positive.pop("action_tag")

    bureau_data = {
        "Experian": {
            "disputes": [account_dispute],
            "goodwill": [account_goodwill],
            "inquiries": [],
            "high_utilization": [account_dispute],
            "all_accounts": [account_dispute, account_positive, account_goodwill],
        }
    }

    disputes_sent = {}
    goodwill_sent = {}
    instructions_capture = {}

    with (
        mock.patch("logic.letter_generator.call_gpt_dispute_letter") as mock_d,
        mock.patch(
            "logic.letter_generator.render_dispute_letter_html", return_value="html"
        ),
        mock.patch(
            "logic.generate_goodwill_letters.goodwill_prompting.generate_goodwill_letter_draft"
        ) as mock_g,
        mock.patch("logic.pdf_renderer.render_html_to_pdf"),
        mock.patch(
            "logic.instructions_generator.render_pdf_from_html",
            side_effect=lambda html, p: instructions_capture.setdefault("html", html),
        ),
        mock.patch(
            "logic.instruction_data_preparation.generate_account_action",
            return_value="Follow advice",
        ),
        mock.patch(
            "logic.instructions_generator.run_compliance_pipeline",
            lambda html, state, session_id, doc_type, ai_client=None: html,
        ),
        mock.patch(
            "logic.instructions_generator.build_instruction_html", return_value="html"
        ),
        mock.patch("logic.instructions_generator.save_json_output"),
        mock.patch(
            "logic.letter_generator.generate_strategy",
            return_value={"dispute_items": {"1": {}}},
        ),
        mock.patch(
            "logic.letter_generator.sanitize_disputes",
            return_value=(False, False, set(), False),
        ),
    ):

        def _d(*args, **kwargs):
            b = args[1]
            d = args[2]
            disputes_sent[b] = d
            return {
                "opening_paragraph": "",
                "accounts": [],
                "inquiries": [],
                "closing_paragraph": "",
            }

        def _g(*args, **kwargs):
            creditor = args[1]
            accounts = args[2]
            goodwill_sent[creditor] = accounts
            return {"intro_paragraph": "", "accounts": [], "closing_paragraph": ""}, []

        mock_d.side_effect = _d
        mock_g.side_effect = _g

        out_dir = Path("output/test_local")
        fake = FakeAIClient()
        generate_dispute_letters_for_all_bureaus(
            client_info, bureau_data, out_dir, False, None, ai_client=fake
        )
        generate_goodwill_letters(
            client_info, bureau_data, out_dir, None, ai_client=fake
        )
        generate_instruction_file(
            client_info, bureau_data, False, out_dir, ai_client=fake
        )
        context, accounts_list = instructions_generator.prepare_instruction_data(
            client_info,
            bureau_data,
            False,
            "2024-01-01",
            "",
            ai_client=fake,
            strategy=None,
        )
        html_content = instructions_generator.build_instruction_html(context)
        instructions_generator.run_compliance_pipeline(
            html_content,
            client_info.get("state"),
            client_info.get("session_id", ""),
            "instructions",
            ai_client=fake,
        )

    assert [a.name for a in disputes_sent.get("Experian", [])] == ["Bank A"]
    assert "Card B" in goodwill_sent

    assert len(accounts_list) == 2
    assert "Unknown" not in html_content


def test_skip_goodwill_when_identity_theft():
    import tempfile
    from orchestrators import run_credit_repair_process

    client_info = {
        "name": "Jane Test",
        "email": "jane@example.com",
        "session_id": "test",
    }
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        proofs = {"smartcredit_report": tmp.name}
        with (
            mock.patch.dict(
                os.environ,
                {
                    "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", "test"),
                    "SMTP_SERVER": "x",
                    "SMTP_PORT": "587",
                    "SMTP_USERNAME": "x",
                    "SMTP_PASSWORD": "x",
                },
            ),
            mock.patch(
                "orchestrators.process_client_intake", return_value=("session", {}, {})
            ),
            mock.patch("orchestrators.classify_client_responses", return_value={}),
            mock.patch(
                "orchestrators.analyze_credit_report",
                return_value=(
                    Path(tmp.name),
                    {},
                    {"Experian": {}},
                    Path(tmp.name).parent,
                ),
            ),
            mock.patch("orchestrators.generate_strategy_plan", return_value={}),
            mock.patch(
                "services.ai_client.build_ai_client", return_value=FakeAIClient()
            ),
            mock.patch(
                "logic.letter_generator.generate_dispute_letters_for_all_bureaus"
            ),
            mock.patch(
                "logic.generate_goodwill_letters.generate_goodwill_letters"
            ) as mock_goodwill,
            mock.patch("logic.generate_custom_letters.generate_custom_letters"),
            mock.patch("logic.instructions_generator.generate_instruction_file"),
            mock.patch("logic.utils.pdf_ops.convert_txts_to_pdfs"),
            mock.patch("orchestrators.send_email_with_attachment"),
            mock.patch("orchestrators.save_analytics_snapshot"),
            mock.patch("logic.bootstrap.extract_all_accounts", return_value=[]),
            mock.patch(
                "logic.upload_validator.move_uploaded_file", return_value=Path(tmp.name)
            ),
            mock.patch("logic.upload_validator.is_safe_pdf", return_value=True),
            mock.patch("orchestrators.save_log_file"),
            mock.patch("shutil.copyfile"),
        ):
            run_credit_repair_process(client_info, proofs, True)
        assert not mock_goodwill.called
    if os.path.exists(tmp.name):
        os.remove(tmp.name)
