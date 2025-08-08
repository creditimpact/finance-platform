import sys
import types
from pathlib import Path
from unittest import mock

sys.path.append(str(Path(__file__).resolve().parents[1]))

# --- optional stubs (safe to leave even when real packages exist) ---
sys.modules.setdefault("openai", types.SimpleNamespace(OpenAI=lambda *_, **__: None))
sys.modules.setdefault("pdfkit", types.SimpleNamespace(configuration=lambda *_, **__: None,
                                                      from_string=lambda *_, **__: None))
sys.modules.setdefault("dotenv", types.SimpleNamespace(load_dotenv=lambda *_, **__: None))
class _DummyEnv:
    def __init__(self, *_, **__):
        pass
    def get_template(self, *_a, **_k):
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
# -------------------------------------------------------------------

from logic.generate_strategy_report import StrategyGenerator
from tests.helpers.fake_ai_client import FakeAIClient
from logic.letter_generator import generate_dispute_letters_for_all_bureaus
from logic.instructions_generator import generate_instruction_file

def test_full_letter_workflow():
    # Mock client + bureau data
    client_info = {
        "name": "John Doe",
        "email": "john@example.com",
        "session_id": "test_session",
    }
    bureau_data = {
        "Experian": {
            "disputes": [
                {"name": "Bank A", "account_number": "1111", "status": "Chargeoff"},
                {"name": "Credit Card B", "account_number": "2222", "status": "Current"},
            ],
            "goodwill": [],
            "inquiries": [],
            "high_utilization": [],
            "all_accounts": [
                {"name": "Bank A", "account_number": "1111", "bureaus": ["Experian"]},
                {"name": "Credit Card B", "account_number": "2222", "bureaus": ["Experian"]},
            ],
        }
    }

    # Strategy output with one account to dispute and one to ignore
    strategy_result = {
        "overview": "Mock strategy",
        "accounts": [
            {"name": "Bank A", "account_number": "1111", "recommendation": "Dispute"},
            {"name": "Credit Card B", "account_number": "2222", "recommendation": "None"},
        ],
        "global_recommendations": ["Keep balances low"],
    }

    letters_created = {}
    instructions_capture = {}

    with (
        mock.patch.object(StrategyGenerator, "generate", return_value=strategy_result),
        mock.patch("logic.letter_generator.render_dispute_letter_html", return_value="html"),
        mock.patch("logic.letter_generator.call_gpt_dispute_letter") as mock_gpt_call,
        mock.patch("logic.letter_generator.render_html_to_pdf"),
        mock.patch("logic.instructions_generator.render_pdf_from_html",
                   side_effect=lambda html, p: instructions_capture.setdefault("html", html)),
        mock.patch("logic.instructions_generator.generate_account_action", return_value="Action"),
        mock.patch("logic.instructions_generator.run_compliance_pipeline", lambda html, state, session_id, doc_type, ai_client=None: html),
        mock.patch("logic.instructions_generator.build_instruction_html", return_value="html"),
        mock.patch("logic.instructions_generator.save_json_output"),
        mock.patch("logic.letter_generator.generate_strategy", return_value={"dispute_items": {"a": {}}}),
        mock.patch("logic.letter_generator.sanitize_disputes", return_value=(False, False, set(), False)),
    ):
        # Capture which disputes were sent to GPT
        def fake_gpt(*args, **kwargs):
            bureau = args[1]
            disputes = args[2]
            letters_created[bureau] = disputes
            return {"opening_paragraph": "", "accounts": [], "inquiries": [], "closing_paragraph": ""}
        mock_gpt_call.side_effect = fake_gpt

        # Generate strategy and merge into bureau data
        from logic.constants import normalize_action_tag

        generator = StrategyGenerator(ai_client=FakeAIClient())
        strategy = generator.generate(client_info, bureau_data, audit=None)
        index = {(acc["name"].lower(), acc["account_number"]): acc for acc in strategy["accounts"]}
        for payload in bureau_data.values():
            for section, items in payload.items():
                if isinstance(items, list):
                    for acc in items:
                        key = (acc.get("name", "").lower(), acc.get("account_number"))
                        if key in index:
                            raw = index[key]["recommendation"]
                            tag, action = normalize_action_tag(raw)
                            acc["action_tag"] = tag or acc.get("action_tag")
                            acc["recommended_action"] = action

        out_dir = Path("output/test_flow")
        fake = FakeAIClient()
        generate_dispute_letters_for_all_bureaus(client_info, bureau_data, out_dir, False, None, ai_client=fake)
        generate_instruction_file(client_info, bureau_data, False, out_dir, strategy=strategy, ai_client=fake)

    # --- Assertions ---
    assert [d["name"] for d in letters_created.get("Experian", [])] == ["Bank A"]
    assert "html" in instructions_capture

