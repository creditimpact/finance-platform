"""Integration-style tests for local workflow."""

# ruff: noqa: E402
import os
import sys
import types
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

import backend.core.logic.rendering.instructions_generator as instructions_generator  # noqa: E402
from backend.core.logic.letters.generate_goodwill_letters import (  # noqa: E402
    generate_goodwill_letters,
)
from backend.core.logic.letters.letter_generator import (  # noqa: E402
    generate_dispute_letters_for_all_bureaus,
)
from backend.core.logic.rendering.instructions_generator import (  # noqa: E402
    generate_instruction_file,
)
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
        mock.patch(
            "backend.core.logic.letters.letter_generator.call_gpt_dispute_letter"
        ) as mock_d,
        mock.patch(
            "backend.core.logic.letters.letter_generator.render_dispute_letter_html",
            return_value="html",
        ),
        mock.patch(
            "backend.core.logic.letters.generate_goodwill_letters.goodwill_prompting.generate_goodwill_letter_draft"
        ) as mock_g,
        mock.patch("backend.core.logic.rendering.pdf_renderer.render_html_to_pdf"),
        mock.patch(
            "backend.core.logic.rendering.instructions_generator.render_pdf_from_html",
            side_effect=lambda html, p: instructions_capture.setdefault("html", html),
        ),
        mock.patch(
            "backend.core.logic.rendering.instruction_data_preparation.generate_account_action",
            return_value="Follow advice",
        ),
        mock.patch(
            "backend.core.logic.rendering.instructions_generator.run_compliance_pipeline",
            lambda html, state, session_id, doc_type, ai_client=None: html,
        ),
        mock.patch(
            "backend.core.logic.rendering.instructions_generator.build_instruction_html",
            return_value="html",
        ),
        mock.patch(
            "backend.core.logic.rendering.instructions_generator.save_json_output"
        ),
        mock.patch(
            "backend.core.logic.letters.letter_generator.generate_strategy",
            return_value={"dispute_items": {"1": {}}},
        ),
        mock.patch(
            "backend.core.logic.letters.letter_generator.sanitize_disputes",
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
        from backend.core.logic.strategy.summary_classifier import ClassificationRecord
        from backend.core.models import BureauPayload, ClientInfo

        client = ClientInfo.from_dict(client_info)
        bureau_models = {k: BureauPayload.from_dict(v) for k, v in bureau_data.items()}
        classification_map = {"1": ClassificationRecord({}, {"category": "late"}, "")}

        generate_dispute_letters_for_all_bureaus(
            client,
            bureau_models,
            out_dir,
            False,
            None,
            ai_client=fake,
            classification_map=classification_map,
        )
        generate_goodwill_letters(
            client,
            bureau_models,
            out_dir,
            None,
            ai_client=fake,
            classification_map=classification_map,
            strategy={"accounts": [account_dispute, account_goodwill]},
        )
        generate_instruction_file(client, bureau_models, False, out_dir, ai_client=fake)
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

    from backend.core.models import ClientInfo, ProofDocuments
    from backend.core.orchestrators import run_credit_repair_process

    client_info = ClientInfo.from_dict(
        {
            "name": "Jane Test",
            "email": "jane@example.com",
            "session_id": "test",
        }
    )
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        proofs = ProofDocuments.from_dict({"smartcredit_report": tmp.name})
        stage_2_5 = {}
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
            mock.patch("orchestrators.generate_strategy_plan", return_value={}),
            mock.patch(
                "services.ai_client.build_ai_client", return_value=FakeAIClient()
            ),
            mock.patch(
                "backend.core.logic.letters.letter_generator.generate_dispute_letters_for_all_bureaus"
            ),
            mock.patch(
                "backend.core.logic.letters.generate_goodwill_letters.generate_goodwill_letters"
            ) as mock_goodwill,
            mock.patch(
                "backend.core.logic.letters.generate_custom_letters.generate_custom_letters"
            ),
            mock.patch(
                "backend.core.logic.rendering.instructions_generator.generate_instruction_file"
            ),
            mock.patch("backend.core.logic.utils.pdf_ops.convert_txts_to_pdfs"),
            mock.patch("orchestrators.send_email_with_attachment"),
            mock.patch("orchestrators.save_analytics_snapshot"),
            mock.patch(
                "backend.core.logic.utils.bootstrap.extract_all_accounts",
                return_value=[],
            ),
            mock.patch(
                "backend.core.logic.compliance.upload_validator.move_uploaded_file",
                return_value=Path(tmp.name),
            ),
            mock.patch(
                "backend.core.logic.compliance.upload_validator.is_safe_pdf",
                return_value=True,
            ),
            mock.patch("orchestrators.save_log_file"),
            mock.patch("shutil.copyfile"),
            mock.patch(
                "orchestrators.analyze_credit_report",
                return_value=(
                    Path(tmp.name),
                    {"negative_accounts": [{"account_id": "1"}]},
                    {"Experian": {}},
                    Path(tmp.name).parent,
                ),
            ),
            mock.patch(
                "orchestrators.update_session",
                side_effect=lambda s, **k: stage_2_5.update(k.get("stage_2_5", {}))
                or {},
            ),
        ):
            run_credit_repair_process(client_info, proofs, True)
        assert not mock_goodwill.called
        assert stage_2_5["1"]["legal_safe_summary"] == "No statement provided"
    if os.path.exists(tmp.name):
        os.remove(tmp.name)


def test_stage_2_5_data_propagates_to_strategy():
    import tempfile

    from backend.core.logic.strategy.generate_strategy_report import StrategyGenerator
    from backend.core.models import ClientInfo, ProofDocuments
    from backend.core.orchestrators import run_credit_repair_process

    client_info = ClientInfo.from_dict(
        {
            "name": "Jane Test",
            "email": "jane@example.com",
            "session_id": "test",
        }
    )
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        proofs = ProofDocuments.from_dict({"smartcredit_report": tmp.name})
        captured = {}
        stage_2_5 = {}

        def fake_generate_letters(*args, **kwargs):
            captured["strategy"] = args[5]

        def fake_save_report(
            self, report, client_info, run_date, base_dir="Clients", stage_2_5_data=None
        ):
            if stage_2_5_data:
                for acc in report.get("accounts", []):
                    acc_id = str(acc.get("account_id", ""))
                    data = stage_2_5_data.get(acc_id, {})
                    acc.setdefault("legal_safe_summary", data.get("legal_safe_summary"))
                    acc.setdefault(
                        "suggested_dispute_frame",
                        data.get("suggested_dispute_frame", ""),
                    )
                    acc.setdefault("rule_hits", data.get("rule_hits", []))
                    acc.setdefault("needs_evidence", data.get("needs_evidence", []))
                    acc.setdefault("red_flags", data.get("red_flags", []))
            return Path(tmp.name)

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
                "services.ai_client.build_ai_client", return_value=FakeAIClient()
            ),
            mock.patch(
                "orchestrators.generate_letters", side_effect=fake_generate_letters
            ),
            mock.patch("orchestrators.finalize_outputs"),
            mock.patch(
                "orchestrators.analyze_credit_report",
                return_value=(
                    Path(tmp.name),
                    {
                        "negative_accounts": [
                            {
                                "account_id": "1",
                                "identity_theft": True,
                                "has_id_theft_affidavit": False,
                                "user_statement_raw": "This is not my account",
                            }
                        ],
                        "all_accounts": [
                            {
                                "account_id": "1",
                                "identity_theft": True,
                                "has_id_theft_affidavit": False,
                                "user_statement_raw": "This is not my account",
                            }
                        ],
                    },
                    {"Experian": {}},
                    Path(tmp.name).parent,
                ),
            ),
            mock.patch(
                "orchestrators.update_session",
                side_effect=lambda s, **k: stage_2_5.update(k.get("stage_2_5", {}))
                or {},
            ),
            mock.patch.object(
                StrategyGenerator,
                "generate",
                return_value={
                    "overview": "",
                    "accounts": [{"account_id": "1", "name": "Acc"}],
                    "global_recommendations": [],
                },
            ),
            mock.patch.object(StrategyGenerator, "save_report", fake_save_report),
        ):
            run_credit_repair_process(client_info, proofs, True)
        acc = captured["strategy"]["accounts"][0]
        assert acc["rule_hits"] == ["E_IDENTITY", "E_IDENTITY_NEEDS_AFFIDAVIT"]
        assert acc["needs_evidence"] == ["identity_theft_affidavit"]
        assert acc["suggested_dispute_frame"] == "fraud"
        assert acc["red_flags"] == []
        assert acc["legal_safe_summary"] == "This is not my account"
    if os.path.exists(tmp.name):
        os.remove(tmp.name)
