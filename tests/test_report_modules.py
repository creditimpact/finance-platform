import importlib
import sys
import types
from pathlib import Path

import pytest

# Ensure tests/helpers is importable
sys.path.append(str(Path(__file__).resolve().parents[1]))
from tests.helpers.fake_ai_client import FakeAIClient


def test_extract_text_from_pdf_calls_pdf_ops(monkeypatch):
    called = {}
    utils_pkg = types.ModuleType("backend.core.logic.utils")
    utils_pkg.__path__ = [
        str(
            Path(__file__).resolve().parents[1] / "backend" / "core" / "logic" / "utils"
        )
    ]
    sys.modules["backend.core.logic.utils"] = utils_pkg

    fake_pdf_ops = types.ModuleType("backend.core.logic.utils.pdf_ops")

    def fake_extract(path, max_chars):
        called["path"] = path
        called["max_chars"] = max_chars
        return "dummy text"

    fake_pdf_ops.extract_pdf_text_safe = fake_extract
    sys.modules["backend.core.logic.utils.pdf_ops"] = fake_pdf_ops
    report_parsing = importlib.import_module(
        "backend.core.logic.report_analysis.report_parsing"
    )

    text = report_parsing.extract_text_from_pdf("report.pdf")
    assert text == "dummy text"
    assert isinstance(called["path"], Path)
    assert called["path"].name == "report.pdf"
    assert called["max_chars"] == 150000


def test_call_ai_analysis_parses_json(tmp_path):
    utils_pkg = types.ModuleType("backend.core.logic.utils")
    utils_pkg.__path__ = [
        str(
            Path(__file__).resolve().parents[1] / "backend" / "core" / "logic" / "utils"
        )
    ]
    sys.modules["backend.core.logic.utils"] = utils_pkg

    fake_pdf_ops = types.ModuleType("backend.core.logic.utils.pdf_ops")
    fake_pdf_ops.extract_pdf_text_safe = lambda *a, **k: ""  # avoid fitz import
    sys.modules.setdefault("backend.core.logic.utils.pdf_ops", fake_pdf_ops)
    report_prompting = importlib.import_module(
        "backend.core.logic.report_analysis.report_prompting"
    )

    client = FakeAIClient()
    client.add_chat_response('{"inquiries": [], "all_accounts": []}')
    out = tmp_path / "result.json"
    data = report_prompting.call_ai_analysis(
        "text", False, out, ai_client=client, strategic_context="goal"
    )
    assert data["inquiries"] == []
    assert out.with_name(out.stem + "_raw.txt").exists()


def test_sanitize_late_counts_removes_unrealistic():
    utils_pkg = types.ModuleType("backend.core.logic.utils")
    utils_pkg.__path__ = [
        str(
            Path(__file__).resolve().parents[1] / "backend" / "core" / "logic" / "utils"
        )
    ]
    sys.modules["backend.core.logic.utils"] = utils_pkg

    report_postprocessing = importlib.import_module(
        "backend.core.logic.report_analysis.report_postprocessing"
    )
    history = {
        "acc": {"Experian": {"30": 2, "60": 13}, "Equifax": {"90": 1}},
        "acc2": {"TU": {"60": 14}},
    }
    report_postprocessing._sanitize_late_counts(history)
    assert history == {"acc": {"Experian": {"30": 2}, "Equifax": {"90": 1}}}


def test_merge_parser_inquiries():
    utils_pkg = types.ModuleType("backend.core.logic.utils")
    utils_pkg.__path__ = [
        str(
            Path(__file__).resolve().parents[1] / "backend" / "core" / "logic" / "utils"
        )
    ]
    sys.modules["backend.core.logic.utils"] = utils_pkg

    report_postprocessing = importlib.import_module(
        "backend.core.logic.report_analysis.report_postprocessing"
    )
    result = {
        "inquiries": [
            {"creditor_name": "Cap One", "date": "01/2024", "bureau": "Experian"}
        ]
    }
    parsed = [
        {"creditor_name": "Cap One", "date": "01/2024", "bureau": "Experian"},
        {"creditor_name": "Chase", "date": "02/2024", "bureau": "TransUnion"},
    ]
    report_postprocessing._merge_parser_inquiries(result, parsed)
    assert len(result["inquiries"]) == 2
    assert any(
        i.get("advisor_comment")
        for i in result["inquiries"]
        if i["creditor_name"] == "Chase"
    )


@pytest.mark.parametrize("identity_theft", [True, False])
def test_analyze_report_wrapper(monkeypatch, tmp_path, identity_theft):
    utils_pkg = types.ModuleType("backend.core.logic.utils")
    utils_pkg.__path__ = [
        str(
            Path(__file__).resolve().parents[1] / "backend" / "core" / "logic" / "utils"
        )
    ]
    sys.modules["backend.core.logic.utils"] = utils_pkg

    fake_pdf_ops = types.ModuleType("backend.core.logic.utils.pdf_ops")
    fake_pdf_ops.extract_pdf_text_safe = lambda *a, **k: "text"
    sys.modules["backend.core.logic.utils.pdf_ops"] = fake_pdf_ops

    report_parsing = importlib.import_module(
        "backend.core.logic.report_analysis.report_parsing"
    )
    report_prompting = importlib.import_module(
        "backend.core.logic.report_analysis.report_prompting"
    )
    report_postprocessing = importlib.import_module(
        "backend.core.logic.report_analysis.report_postprocessing"
    )
    analyze_report = importlib.import_module(
        "backend.core.logic.report_analysis.analyze_report"
    )

    client = FakeAIClient()
    response = '{"summary_metrics": {"total_inquiries": 0}, "all_accounts": []}'
    client.add_chat_response(response)
    client.add_chat_response(response)

    # Parsing stage
    monkeypatch.setattr(report_parsing, "extract_text_from_pdf", lambda p: "text")
    monkeypatch.setattr(analyze_report, "extract_inquiries", lambda text: [])
    monkeypatch.setattr(
        report_prompting,
        "extract_late_history_blocks",
        lambda text, return_raw_map=False: ({}, {}) if return_raw_map else {},
    )
    monkeypatch.setattr(report_prompting, "extract_inquiries", lambda text: [])

    # Post-processing no-ops
    monkeypatch.setattr(
        report_postprocessing, "_sanitize_late_counts", lambda hist: None
    )
    monkeypatch.setattr(
        report_postprocessing, "_cleanup_unverified_late_text", lambda res, ver: None
    )
    monkeypatch.setattr(
        report_postprocessing,
        "_inject_missing_late_accounts",
        lambda res, hist, raw: None,
    )
    monkeypatch.setattr(
        report_postprocessing, "_merge_parser_inquiries", lambda res, parsed: None
    )
    monkeypatch.setattr(
        report_postprocessing, "_reconcile_account_headings", lambda res, headings: None
    )
    monkeypatch.setattr(
        report_postprocessing, "validate_analysis_sanity", lambda res: []
    )

    default_goal = (
        "Improve credit score significantly within the next 3-6 months using strategies such as authorized users, "
        "credit building tools, and removal of negative items."
    )

    baseline = report_prompting.call_ai_analysis(
        "text",
        identity_theft,
        tmp_path / "baseline.json",
        ai_client=client,
        strategic_context=default_goal,
    )

    result = analyze_report.analyze_credit_report(
        tmp_path / "dummy.pdf",
        tmp_path / "out.json",
        {"goal": "improve credit", "is_identity_theft": identity_theft},
        ai_client=client,
    )
    assert result == baseline


def test_analyze_credit_report_skips_ai(monkeypatch, tmp_path):
    from backend.core.logic.report_analysis import (
        analyze_report,
        report_parsing,
        report_prompting,
    )

    def boom(*a, **k):  # pragma: no cover - fail if called
        raise AssertionError("AI should not be called")

    monkeypatch.setattr(report_prompting, "call_ai_analysis", boom)
    monkeypatch.setattr(report_parsing, "extract_text_from_pdf", lambda p: "dummy text")

    result = analyze_report.analyze_credit_report(
        tmp_path / "dummy.pdf",
        tmp_path / "out.json",
        {},
        ai_client=None,
        run_ai=False,
    )
    assert result["negative_accounts"] == []
