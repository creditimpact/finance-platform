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
    types.SimpleNamespace(configuration=lambda *_, **__: None, from_string=lambda *_, **__: None),
)
sys.modules.setdefault("dotenv", types.SimpleNamespace(load_dotenv=lambda *_, **__: None))
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
    types.SimpleNamespace(Environment=_DummyEnv, FileSystemLoader=lambda *_, **__: None),
)
sys.modules.setdefault("fpdf", types.SimpleNamespace(FPDF=object))
sys.modules.setdefault("pdfplumber", types.SimpleNamespace(open=lambda *_, **__: None))
sys.modules.setdefault("fitz", types.SimpleNamespace(open=lambda *_, **__: None))

from logic.letter_generator import generate_dispute_letters_for_all_bureaus
from logic.generate_goodwill_letters import generate_goodwill_letters
from logic.instructions_generator import generate_instruction_file, generate_html


def run_checks():
    client_info = {"name": "Jane Test", "email": "jane@example.com", "session_id": "test"}

    account_dispute = {
        "name": "Bank A",
        "account_number": "1234",
        "status": "Chargeoff",
        "bureaus": ["Experian"],
        "action_tag": "dispute",
        "recommended_action": "Dispute",
        "opened_date": "01/01/2020",
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
        mock.patch("logic.letter_generator.render_html_to_pdf"),
        mock.patch("logic.generate_goodwill_letters.call_gpt_for_goodwill_letter") as mock_g,
        mock.patch("logic.generate_goodwill_letters.render_html_to_pdf"),
        mock.patch("logic.instructions_generator.render_pdf_from_html", side_effect=lambda html, p: instructions_capture.setdefault("html", html)),
        mock.patch("logic.instructions_generator.generate_account_action", return_value="Follow advice"),
        mock.patch("logic.instructions_generator.save_json_output"),
    ):
        def _d(ci, b, d, i, t):
            disputes_sent[b] = d
            return {"opening_paragraph": "", "accounts": [], "inquiries": [], "closing_paragraph": ""}

        def _g(*args, **kwargs):
            creditor = args[1]
            accounts = args[2]
            goodwill_sent[creditor] = accounts
            return {"intro_paragraph": "", "accounts": [], "closing_paragraph": ""}

        mock_d.side_effect = _d
        mock_g.side_effect = _g

        out_dir = Path("output/test_local")
        generate_dispute_letters_for_all_bureaus(client_info, bureau_data, out_dir, False)
        generate_goodwill_letters(client_info, bureau_data, out_dir)
        generate_instruction_file(client_info, bureau_data, False, out_dir)
        html_content, accounts_list = generate_html(
            client_info,
            bureau_data,
            False,
            "2024-01-01",
            "",
            None,
        )

    assert [a["name"] for a in disputes_sent.get("Experian", [])] == ["Bank A"]
    assert "Card B" in goodwill_sent

    assert len(accounts_list) == 2
    assert "Unknown" not in html_content
    print("✅ All checks passed")


def test_skip_goodwill_when_identity_theft():
    import tempfile
    from main import run_credit_repair_process

    client_info = {"name": "Jane Test", "email": "jane@example.com", "session_id": "test"}
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
            mock.patch("main.extract_bureau_info_column_refined", return_value={"data": {}}),
            mock.patch("main.analyze_credit_report", return_value={}),
            mock.patch("main.gather_supporting_docs_text", return_value=""),
            mock.patch("main.StrategyGenerator") as mock_strat,
            mock.patch("main.generate_dispute_letters_for_all_bureaus"),
            mock.patch("main.generate_goodwill_letters") as mock_goodwill,
            mock.patch("main.generate_custom_letters"),
            mock.patch("main.generate_instruction_file"),
            mock.patch("main.convert_txts_to_pdfs"),
            mock.patch("main.send_email_with_attachment"),
            mock.patch("main.save_analytics_snapshot"),
            mock.patch("main.extract_all_accounts", return_value=[]),
            mock.patch("logic.upload_validator.move_uploaded_file", return_value=Path(tmp.name)),
            mock.patch("logic.upload_validator.is_safe_pdf", return_value=True),
            mock.patch("main.save_log_file"),
            mock.patch("main.copyfile"),
        ):
            mock_strat.return_value.generate.return_value = {}
            mock_strat.return_value.save_report.return_value = None
            run_credit_repair_process(client_info, proofs, True)
        assert not mock_goodwill.called
    if os.path.exists(tmp.name):
        os.remove(tmp.name)


if __name__ == "__main__":
    run_checks()
    test_skip_goodwill_when_identity_theft()
