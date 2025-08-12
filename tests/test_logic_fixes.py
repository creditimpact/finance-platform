"""Regression tests for logic modules."""

# ruff: noqa: E402
import sys
import types
import json
from pathlib import Path
from unittest import mock
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

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

    def get_template(self, *_, **__):
        class _T:
            def render(self, **_):
                return ""

        return _T()


sys.modules.setdefault("fpdf", types.SimpleNamespace(FPDF=object))
sys.modules.setdefault("pdfplumber", types.SimpleNamespace(open=lambda *_, **__: None))
sys.modules.setdefault("fitz", types.SimpleNamespace(open=lambda *_, **__: None))

import backend.core.logic.instructions_generator as instructions_generator
from backend.core.logic.generate_goodwill_letters import generate_goodwill_letters
from backend.core.logic.letter_generator import generate_dispute_letters_for_all_bureaus
from tests.helpers.fake_ai_client import FakeAIClient
from backend.core.logic.process_accounts import process_analyzed_report
from backend.core.logic.utils.text_parsing import (
    extract_late_history_blocks,
    extract_account_blocks,
)


def test_dedup_without_numbers():
    bureau_data = {
        "Experian": {
            "disputes": [],
            "goodwill": [],
            "inquiries": [],
            "high_utilization": [],
            "all_accounts": [
                {
                    "name": "Capital One",
                    "account_number": "",
                    "status": "Open",
                    "bureaus": ["Experian"],
                },
                {
                    "name": "CAP ONE",
                    "account_number": None,
                    "status": "Open",
                    "bureaus": ["Experian"],
                },
            ],
        }
    }
    with (
        mock.patch(
            "logic.instruction_data_preparation.generate_account_action",
            return_value="A",
        ),
    ):
        fake = FakeAIClient()
        context, accounts = instructions_generator.prepare_instruction_data(
            {"name": "Test"},
            bureau_data,
            False,
            "2024-01-01",
            "",
            ai_client=fake,
            strategy=None,
        )
        html = instructions_generator.build_instruction_html(context)
        instructions_generator.run_compliance_pipeline(
            html,
            None,
            "",
            "instructions",
            ai_client=fake,
        )
    assert len(accounts) == 1
    print("dedup ok")


def test_inquiry_matching():
    sample = {
        "negative_accounts": [],
        "open_accounts_with_issues": [],
        "high_utilization_accounts": [],
        "account_inquiry_matches": [],
        "all_accounts": [{"name": "Bank of America", "bureaus": ["Experian"]}],
        "inquiries": [
            {"creditor_name": "BK OF AMER", "bureau": "Experian"},
            {"creditor_name": "Some Store", "bureau": "Experian"},
        ],
    }
    path = Path("output/tmp_report.json")
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(sample))
    output = process_analyzed_report(path)
    inqs = output["Experian"]["inquiries"]
    assert len(inqs) == 1 and inqs[0]["creditor_name"] == "Some Store"
    print("inquiry ok")


def test_goodwill_generation():
    bureau_data = {
        "Experian": {
            "disputes": [],
            "goodwill": [
                {
                    "name": "Card Co",
                    "bureaus": ["Experian"],
                    "late_payments": {"Experian": {"30": 1}},
                    "status": "Closed",
                }
            ],
            "inquiries": [],
            "high_utilization": [],
            "all_accounts": [],
        }
    }
    sent = {}
    with (
        mock.patch(
            "logic.generate_goodwill_letters.goodwill_prompting.generate_goodwill_letter_draft"
        ) as mock_g,
        mock.patch("logic.pdf_renderer.render_html_to_pdf"),
    ):
        out_dir = Path("output/tmp")
        out_dir.mkdir(parents=True, exist_ok=True)

        def _cb(*args, **kwargs):
            creditor = args[1] if len(args) > 1 else None
            accounts = args[2] if len(args) > 2 else None
            sent[creditor] = {
                "accounts": accounts,
            }
            return {"intro_paragraph": "", "accounts": [], "closing_paragraph": ""}, []

        mock_g.side_effect = _cb
        fake = FakeAIClient()
        generate_goodwill_letters(
            {"name": "T"},
            bureau_data,
            out_dir,
            None,
            ai_client=fake,
        )
    assert "Card Co" in sent
    print("goodwill ok")


def test_goodwill_on_closed_account():
    bureau_data = {
        "Experian": {
            "disputes": [
                {
                    "name": "Old Card",
                    "bureaus": ["Experian"],
                    "late_payments": {"Experian": {"30": 1}},
                    "status": "Closed",
                    "goodwill_on_closed": True,
                }
            ],
            "goodwill": [],
            "inquiries": [],
            "high_utilization": [],
            "all_accounts": [],
        }
    }
    called = {}
    with (
        mock.patch(
            "logic.generate_goodwill_letters.goodwill_prompting.generate_goodwill_letter_draft"
        ) as mock_g,
        mock.patch("logic.pdf_renderer.render_html_to_pdf"),
    ):
        out_dir = Path("output/tmp2")
        out_dir.mkdir(parents=True, exist_ok=True)

        def _cb(*args, **kwargs):
            creditor = args[1] if len(args) > 1 else None
            accounts = args[2] if len(args) > 2 else None
            called[creditor] = accounts
            return {"intro_paragraph": "", "accounts": [], "closing_paragraph": ""}, []

        mock_g.side_effect = _cb
        fake = FakeAIClient()
        generate_goodwill_letters(
            {"name": "T"}, bureau_data, out_dir, None, ai_client=fake
        )
    assert "Old Card" in called
    print("goodwill closed ok")


def test_skip_goodwill_when_no_late_payments():
    bureau_data = {
        "Experian": {
            "disputes": [],
            "goodwill": [
                {
                    "name": "No Late Card",
                    "bureaus": ["Experian"],
                    "late_payments": {"Experian": {"30": 0, "60": 0, "90": 0}},
                    "status": "Open",
                }
            ],
            "inquiries": [],
            "high_utilization": [],
            "all_accounts": [],
        }
    }
    with (
        mock.patch(
            "logic.generate_goodwill_letters.goodwill_prompting.generate_goodwill_letter_draft"
        ) as mock_g,
        mock.patch("logic.pdf_renderer.render_html_to_pdf"),
    ):
        out_dir = Path("output/tmp3")
        out_dir.mkdir(parents=True, exist_ok=True)
        generate_goodwill_letters(
            {"name": "T"}, bureau_data, out_dir, None, ai_client=FakeAIClient()
        )
    assert not mock_g.called
    print("goodwill skip ok")


def test_skip_goodwill_on_collections():
    bureau_data = {
        "Experian": {
            "disputes": [],
            "goodwill": [
                {
                    "name": "Collection Co",
                    "bureaus": ["Experian"],
                    "late_payments": {"Experian": {"30": 1}},
                    "status": "Collections",
                }
            ],
            "inquiries": [],
            "high_utilization": [],
            "all_accounts": [],
        }
    }
    with (
        mock.patch(
            "logic.generate_goodwill_letters.goodwill_prompting.generate_goodwill_letter_draft"
        ) as mock_g,
        mock.patch("logic.pdf_renderer.render_html_to_pdf"),
    ):
        out_dir = Path("output/tmp4")
        out_dir.mkdir(parents=True, exist_ok=True)
        generate_goodwill_letters(
            {"name": "T"}, bureau_data, out_dir, None, ai_client=FakeAIClient()
        )
    assert not mock_g.called
    print("goodwill collection skip ok")


def test_skip_goodwill_edge_statuses():
    bureau_data = {
        "Experian": {
            "disputes": [],
            "goodwill": [
                {
                    "name": "Repo Co",
                    "bureaus": ["Experian"],
                    "late_payments": {"Experian": {"30": 1}},
                    "status": "Charge Off",
                },
            ],
            "inquiries": [],
            "high_utilization": [],
            "all_accounts": [],
        }
    }
    with (
        mock.patch(
            "logic.generate_goodwill_letters.goodwill_prompting.generate_goodwill_letter_draft"
        ) as mock_g,
        mock.patch("logic.pdf_renderer.render_html_to_pdf"),
    ):
        out_dir = Path("output/tmp4a")
        out_dir.mkdir(parents=True, exist_ok=True)
        generate_goodwill_letters(
            {"name": "T"}, bureau_data, out_dir, None, ai_client=FakeAIClient()
        )
    assert not mock_g.called
    print("goodwill edge skip ok")


def test_fallback_tagging_collections():
    sample = {
        "negative_accounts": [
            {
                "name": "ACME",
                "bureaus": ["Experian"],
                "account_number": "123456",
                "status": "Collections",
            }
        ],
        "open_accounts_with_issues": [],
        "high_utilization_accounts": [],
        "account_inquiry_matches": [],
        "all_accounts": [],
        "inquiries": [],
    }
    path = Path("output/tmp_report2.json")
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(sample))
    output = process_analyzed_report(path)
    acc = output["Experian"]["disputes"][0]
    assert acc.get("action_tag") == "dispute"
    print("fallback tag ok")


def test_fallback_tagging_extra_keywords():
    sample = {
        "negative_accounts": [
            {
                "name": "REPO CO",
                "bureaus": ["Experian"],
                "account_number": "789012",
                "status": "Repossession",
            }
        ],
        "open_accounts_with_issues": [],
        "high_utilization_accounts": [],
        "account_inquiry_matches": [],
        "all_accounts": [],
        "inquiries": [],
    }
    path = Path("output/tmp_report3.json")
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(sample))
    output = process_analyzed_report(path)
    acc = output["Experian"]["disputes"][0]
    assert acc.get("action_tag") == "dispute"
    print("fallback extra ok")


def test_normalize_action_tag_aliases():
    from backend.core.logic.constants import normalize_action_tag

    phrases = [
        "dispute for verification",
        "challenge the debt",
        "request deletion",
        "dispute the accuracy",
        "verify this record",
    ]
    for ph in phrases:
        tag, act = normalize_action_tag(ph)
        assert tag == "dispute" and act == "Dispute"
    print("alias map ok")


def test_letter_duplicate_accounts_removed():
    bureau_data = {
        "Experian": {
            "disputes": [
                {
                    "name": "Bank A",
                    "account_number": "123",
                    "account_id": "1",
                    "action_tag": "dispute",
                },
                {
                    "name": "BANK A",
                    "account_number": "123",
                    "account_id": "2",
                    "action_tag": "dispute",
                },
                {
                    "name": "Other Bank",
                    "account_number": "",
                    "account_id": "3",
                    "action_tag": "dispute",
                },
                {
                    "name": "Other Bank",
                    "account_number": "N/A",
                    "account_id": "4",
                    "action_tag": "dispute",
                },
            ],
            "goodwill": [],
            "inquiries": [],
            "high_utilization": [],
            "all_accounts": [],
        }
    }

    strategy = {"dispute_items": {"1": {}, "2": {}, "3": {}, "4": {}}}
    sent = {}
    with (
        mock.patch("logic.letter_generator.call_gpt_dispute_letter") as mock_d,
        mock.patch("logic.pdf_renderer.render_html_to_pdf"),
        mock.patch(
            "logic.compliance_pipeline.run_compliance_pipeline",
            lambda html, state, session_id, doc_type, ai_client=None: html,
        ),
        mock.patch("logic.letter_generator.generate_strategy", return_value=strategy),
    ):
        out_dir = Path("output/tmp_dupes")
        out_dir.mkdir(parents=True, exist_ok=True)

        def _cb(*args, **kwargs):
            b = args[1] if len(args) > 1 else None
            disputes = args[2] if len(args) > 2 else None
            sent[b] = disputes
            return {
                "opening_paragraph": "",
                "accounts": [],
                "inquiries": [],
                "closing_paragraph": "",
            }

        mock_d.side_effect = _cb
        fake = FakeAIClient()
        with pytest.warns(UserWarning):
            generate_dispute_letters_for_all_bureaus(
                {"name": "T"}, bureau_data, out_dir, False, None, ai_client=fake
            )

    assert len(sent.get("Experian", [])) == 2
    names = {d.name.lower() for d in sent["Experian"]}
    assert names == {"bank a", "other bank"}
    print("letter dupes ok")


def test_partial_account_number_deduplication():
    bureau_data = {
        "Experian": {
            "disputes": [
                {
                    "name": "ACIMA DIGITAL FKA SIMP",
                    "account_number": "XXXX1234",
                    "account_id": "1",
                    "action_tag": "dispute",
                },
                {
                    "name": "ACIMA DIGITAL FKA SIMP",
                    "account_number": "1234",
                    "account_id": "2",
                    "action_tag": "dispute",
                },
            ],
            "goodwill": [],
            "inquiries": [],
            "high_utilization": [],
            "all_accounts": [],
        }
    }
    strategy = {"dispute_items": {"1": {}, "2": {}}}
    sent = {}
    with (
        mock.patch("logic.letter_generator.call_gpt_dispute_letter") as mock_d,
        mock.patch("logic.pdf_renderer.render_html_to_pdf"),
        mock.patch(
            "logic.compliance_pipeline.run_compliance_pipeline",
            lambda html, state, session_id, doc_type, ai_client=None: html,
        ),
        mock.patch("logic.letter_generator.generate_strategy", return_value=strategy),
    ):
        out_dir = Path("output/tmp_dupe_nums")
        out_dir.mkdir(parents=True, exist_ok=True)

        def _cb(*args, **kwargs):
            b = args[1] if len(args) > 1 else None
            disputes = args[2] if len(args) > 2 else None
            sent[b] = disputes
            return {
                "opening_paragraph": "",
                "accounts": [],
                "inquiries": [],
                "closing_paragraph": "",
            }

        mock_d.side_effect = _cb
        fake = FakeAIClient()
        with pytest.warns(UserWarning):
            generate_dispute_letters_for_all_bureaus(
                {"name": "T"}, bureau_data, out_dir, False, None, ai_client=fake
            )

    assert len(sent.get("Experian", [])) == 1
    print("partial dedupe ok")


def test_skip_goodwill_for_disputed_account():
    bureau_data = {
        "Experian": {
            "disputes": [
                {"name": "Card Co", "account_number": "123", "action_tag": "dispute"}
            ],
            "goodwill": [
                {
                    "name": "Card Co",
                    "account_number": "123",
                    "late_payments": {"Experian": {"30": 1}},
                    "status": "Closed",
                }
            ],
            "inquiries": [],
            "high_utilization": [],
            "all_accounts": [],
        }
    }

    with (
        mock.patch(
            "logic.generate_goodwill_letters.goodwill_prompting.generate_goodwill_letter_draft"
        ) as mock_g,
        mock.patch("logic.pdf_renderer.render_html_to_pdf"),
    ):
        out_dir = Path("output/tmp_dupe_skip")
        out_dir.mkdir(parents=True, exist_ok=True)
        generate_goodwill_letters(
            {"name": "T"}, bureau_data, out_dir, None, ai_client=FakeAIClient()
        )

    assert not mock_g.called
    print("goodwill dispute skip ok")


def test_extract_late_history_blocks():
    text = """
    BMW FIN SVC
    TransUnion 30: 1 60: 0 90: 0
    Experian 30:0 60:1 90:0
    """
    result = extract_late_history_blocks(text, {"BMW FIN SVC"})
    assert result == {}
    print("late history ok")


def test_extract_late_history_no_header():
    text = """
    BMW FIN SVC
    TransUnion
    30:1 60:1 90:1
    """
    result = extract_late_history_blocks(text, {"BMW FIN SVC"})
    assert result == {}
    print("late history alt ok")


def test_skip_placeholder_heading():
    text = "CO CO CO\nTransUnion 30:0 60:0 90:0"
    blocks = extract_account_blocks(text)
    assert blocks == []
    print("placeholder skip ok")


def test_account_block_extraction_and_parsing():
    text = """
    ALLY FINCL
    TransUnion 30:1 60:0 90:0
    Experian 30:0 60:0 90:0
    Equifax 30:1 60:1 90:0
    WELLS FARGO
    TransUnion 30:2 60:0 90:0
    Equifax 30:2 60:1 90:1
    """
    blocks = extract_account_blocks(text)
    assert blocks == []
    print("account blocks ok")
