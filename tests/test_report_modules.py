import importlib
import logging
import sys
import types
from pathlib import Path

import pytest

# Ensure tests/helpers is importable
sys.path.append(str(Path(__file__).resolve().parents[1]))
import backend.core.logic.report_analysis.report_prompting as report_prompting
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


def test_call_ai_analysis_parses_json(tmp_path, monkeypatch):
    monkeypatch.setenv("ANALYSIS_DEBUG_STORE_RAW", "1")
    import backend.core.logic.report_analysis.flags as ra_flags

    importlib.reload(ra_flags)

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
    from backend.core.logic.report_analysis import analysis_cache

    analysis_cache.reset_cache()
    ra_flags.FLAGS.debug_store_raw = True

    assert ra_flags.FLAGS.debug_store_raw
    client = FakeAIClient()
    client.add_chat_response('{"inquiries": [], "all_accounts": []}')
    out = tmp_path / "result.json"
    data = report_prompting.call_ai_analysis(
        "text",
        False,
        out,
        ai_client=client,
        strategic_context="goal",
        request_id="req",
        doc_fingerprint="fp1",
    )
    assert data["inquiries"] == []
    assert data["prompt_version"] == report_prompting.ANALYSIS_PROMPT_VERSION
    assert data["schema_version"] == report_prompting.ANALYSIS_SCHEMA_VERSION
    ra_flags.FLAGS.debug_store_raw = False


def test_call_ai_analysis_populates_defaults_and_logs(tmp_path, caplog):
    utils_pkg = types.ModuleType("backend.core.logic.utils")
    utils_pkg.__path__ = [
        str(
            Path(__file__).resolve().parents[1] / "backend" / "core" / "logic" / "utils"
        )
    ]
    sys.modules["backend.core.logic.utils"] = utils_pkg

    fake_pdf_ops = types.ModuleType("backend.core.logic.utils.pdf_ops")
    fake_pdf_ops.extract_pdf_text_safe = lambda *a, **k: ""
    sys.modules.setdefault("backend.core.logic.utils.pdf_ops", fake_pdf_ops)
    report_prompting = importlib.import_module(
        "backend.core.logic.report_analysis.report_prompting"
    )
    from backend.core.logic.report_analysis import analysis_cache

    analysis_cache.reset_cache()

    client = FakeAIClient()
    client.add_chat_response("{}")
    out = tmp_path / "result.json"
    with caplog.at_level(logging.WARNING):
        data = report_prompting.call_ai_analysis(
            "text",
            False,
            out,
            ai_client=client,
            strategic_context="goal",
            request_id="req",
            doc_fingerprint="fp2",
        )
    assert data["negative_accounts"] == []
    assert data["summary_metrics"]["total_inquiries"] == 0
    assert data["prompt_version"] == report_prompting.ANALYSIS_PROMPT_VERSION
    assert data["schema_version"] == report_prompting.ANALYSIS_SCHEMA_VERSION
    assert any(r.__dict__.get("validation_errors") for r in caplog.records)


def test_call_ai_analysis_rejects_non_json(tmp_path, caplog):
    utils_pkg = types.ModuleType("backend.core.logic.utils")
    utils_pkg.__path__ = [
        str(
            Path(__file__).resolve().parents[1] / "backend" / "core" / "logic" / "utils"
        )
    ]
    sys.modules["backend.core.logic.utils"] = utils_pkg

    fake_pdf_ops = types.ModuleType("backend.core.logic.utils.pdf_ops")
    fake_pdf_ops.extract_pdf_text_safe = lambda *a, **k: ""
    sys.modules.setdefault("backend.core.logic.utils.pdf_ops", fake_pdf_ops)
    report_prompting = importlib.import_module(
        "backend.core.logic.report_analysis.report_prompting"
    )
    from backend.core.logic.report_analysis import analysis_cache

    analysis_cache.reset_cache()

    client = FakeAIClient()
    client.add_chat_response("not json")
    out = tmp_path / "result.json"
    with caplog.at_level(logging.WARNING):
        data = report_prompting.call_ai_analysis(
            "text",
            False,
            out,
            ai_client=client,
            request_id="req",
            doc_fingerprint="fp3",
        )
    assert data["negative_accounts"] == []
    assert any(r.__dict__.get("validation_errors") for r in caplog.records)


def test_call_ai_analysis_retries_and_succeeds(tmp_path, caplog):
    utils_pkg = types.ModuleType("backend.core.logic.utils")
    utils_pkg.__path__ = [
        str(
            Path(__file__).resolve().parents[1] / "backend" / "core" / "logic" / "utils"
        )
    ]
    sys.modules["backend.core.logic.utils"] = utils_pkg

    fake_pdf_ops = types.ModuleType("backend.core.logic.utils.pdf_ops")
    fake_pdf_ops.extract_pdf_text_safe = lambda *a, **k: ""
    sys.modules.setdefault("backend.core.logic.utils.pdf_ops", fake_pdf_ops)
    report_prompting = importlib.import_module(
        "backend.core.logic.report_analysis.report_prompting"
    )
    from backend.core.logic.report_analysis import analysis_cache

    analysis_cache.reset_cache()

    client = FakeAIClient()
    client.add_chat_response("not json")
    client.add_chat_response('{"inquiries": [], "all_accounts": []}')
    out = tmp_path / "result.json"
    with caplog.at_level(logging.INFO):
        data = report_prompting.call_ai_analysis(
            "text",
            False,
            out,
            ai_client=client,
            request_id="req",
            doc_fingerprint="fp_retry",
        )
    assert data["inquiries"] == []
    attempts = [r for r in caplog.records if r.__dict__.get("bureau")]
    assert any(r.__dict__.get("attempt") == 1 for r in attempts)
    assert any(r.__dict__.get("attempt") == 2 for r in attempts)


def test_call_ai_analysis_logs_low_recall(tmp_path, caplog, monkeypatch):
    utils_pkg = types.ModuleType("backend.core.logic.utils")
    utils_pkg.__path__ = [
        str(
            Path(__file__).resolve().parents[1] / "backend" / "core" / "logic" / "utils"
        )
    ]
    sys.modules["backend.core.logic.utils"] = utils_pkg

    fake_pdf_ops = types.ModuleType("backend.core.logic.utils.pdf_ops")
    fake_pdf_ops.extract_pdf_text_safe = lambda *a, **k: ""
    sys.modules.setdefault("backend.core.logic.utils.pdf_ops", fake_pdf_ops)
    report_prompting = importlib.import_module(
        "backend.core.logic.report_analysis.report_prompting"
    )
    from backend.core.logic.report_analysis import analysis_cache

    analysis_cache.reset_cache()

    client = FakeAIClient()
    client.add_chat_response('{"all_accounts": [{"name": "Cap One"}]}')
    client.add_chat_response(
        '{"all_accounts": [{"name": "Cap One"}, {"name": "Chase"}]}'
    )

    monkeypatch.setattr(
        report_prompting,
        "extract_account_headings",
        lambda text: [("cap one", "Cap One"), ("chase", "Chase")],
    )
    monkeypatch.setattr(
        report_prompting,
        "extract_late_history_blocks",
        lambda text, return_raw_map=False: ({}, {}) if return_raw_map else {},
    )
    monkeypatch.setattr(report_prompting, "extract_inquiries", lambda text: [])
    monkeypatch.setattr(
        report_prompting, "normalize_creditor_name", lambda n: (n or "").lower()
    )
    monkeypatch.setattr(
        report_prompting, "normalize_creditor_name", lambda n: (n or "").lower()
    )
    monkeypatch.setattr(
        report_prompting, "normalize_creditor_name", lambda n: (n or "").lower()
    )

    out = tmp_path / "result.json"
    with caplog.at_level(logging.INFO):
        data = report_prompting.call_ai_analysis(
            "text",
            False,
            out,
            ai_client=client,
            request_id="req",
            doc_fingerprint="fp_low",
        )
    assert len(data["all_accounts"]) == 2
    attempts = [r for r in caplog.records if r.__dict__.get("bureau")]
    assert any(r.__dict__.get("attempt") == 1 for r in attempts)
    assert not any(r.__dict__.get("attempt") == 2 for r in attempts)
    assert any(
        "LOW_RECALL" in (r.__dict__.get("validation_errors") or [])
        for r in caplog.records
    )
    assert data["bureau_sections"][0].get("warnings") == ["LOW_RECALL"]
    assert any(
        "Chase" in (r.__dict__.get("unmatched_headings") or []) for r in caplog.records
    )


def test_low_recall_does_not_clear_other_bureaus(tmp_path, monkeypatch):
    utils_pkg = types.ModuleType("backend.core.logic.utils")
    utils_pkg.__path__ = [
        str(
            Path(__file__).resolve().parents[1] / "backend" / "core" / "logic" / "utils"
        )
    ]
    sys.modules["backend.core.logic.utils"] = utils_pkg

    fake_pdf_ops = types.ModuleType("backend.core.logic.utils.pdf_ops")
    fake_pdf_ops.extract_pdf_text_safe = lambda *a, **k: ""
    sys.modules.setdefault("backend.core.logic.utils.pdf_ops", fake_pdf_ops)
    report_prompting = importlib.import_module(
        "backend.core.logic.report_analysis.report_prompting"
    )
    from backend.core.logic.report_analysis import analysis_cache

    analysis_cache.reset_cache()

    client = FakeAIClient()
    client.add_chat_response('{"all_accounts": [{"name": "Cap One"}]}')
    client.add_chat_response('{"all_accounts": [{"name": "BofA"}]}')

    def fake_headings(text):
        if "Experian" in text:
            return [("cap one", "Cap One"), ("chase", "Chase")]
        return [("bofa", "BofA")]

    monkeypatch.setattr(report_prompting, "extract_account_headings", fake_headings)
    monkeypatch.setattr(
        report_prompting,
        "extract_late_history_blocks",
        lambda text, return_raw_map=False: ({}, {}) if return_raw_map else {},
    )
    monkeypatch.setattr(report_prompting, "extract_inquiries", lambda text: [])
    monkeypatch.setattr(
        report_prompting, "normalize_creditor_name", lambda n: (n or "").lower()
    )

    out = tmp_path / "result.json"
    text = "Experian section Equifax section"
    data = report_prompting.call_ai_analysis(
        text,
        False,
        out,
        ai_client=client,
        request_id="req",
        doc_fingerprint="fp_multi_low",
    )

    names = {acc["name"] for acc in data["all_accounts"]}
    assert names == {"Cap One", "BofA"}
    assert "LOW_RECALL" in data["bureau_sections"][0].get("warnings")


def test_call_ai_analysis_merges_segments(tmp_path):
    utils_pkg = types.ModuleType("backend.core.logic.utils")
    utils_pkg.__path__ = [
        str(
            Path(__file__).resolve().parents[1] / "backend" / "core" / "logic" / "utils"
        )
    ]
    sys.modules["backend.core.logic.utils"] = utils_pkg

    fake_pdf_ops = types.ModuleType("backend.core.logic.utils.pdf_ops")
    fake_pdf_ops.extract_pdf_text_safe = lambda *a, **k: ""
    sys.modules.setdefault("backend.core.logic.utils.pdf_ops", fake_pdf_ops)
    report_prompting = importlib.import_module(
        "backend.core.logic.report_analysis.report_prompting"
    )
    from backend.core.logic.report_analysis import analysis_cache

    analysis_cache.reset_cache()

    client = FakeAIClient()
    client.add_chat_response(
        '{"inquiries":[{"creditor_name":"A","date":"01/2024","bureau":"Experian"}],"all_accounts":[{"name":"Cap One","bureaus":["Experian"]}]}'
    )
    client.add_chat_response(
        '{"inquiries":[{"creditor_name":"A","date":"01/2024","bureau":"Equifax"}],"all_accounts":[{"name":"Cap One","bureaus":["Equifax"]}]}'
    )
    out = tmp_path / "result.json"
    text = "Experian section Equifax section"
    data = report_prompting.call_ai_analysis(
        text,
        False,
        out,
        ai_client=client,
        request_id="req",
        doc_fingerprint="fp4",
    )
    assert len(data["inquiries"]) == 2
    assert sorted(data["all_accounts"][0]["bureaus"]) == [
        "Equifax",
        "Experian",
    ]


def test_call_ai_analysis_adds_confidence_and_flags(tmp_path, monkeypatch):
    utils_pkg = types.ModuleType("backend.core.logic.utils")
    utils_pkg.__path__ = [
        str(
            Path(__file__).resolve().parents[1] / "backend" / "core" / "logic" / "utils"
        )
    ]
    sys.modules["backend.core.logic.utils"] = utils_pkg

    fake_pdf_ops = types.ModuleType("backend.core.logic.utils.pdf_ops")
    fake_pdf_ops.extract_pdf_text_safe = lambda *a, **k: ""
    sys.modules.setdefault("backend.core.logic.utils.pdf_ops", fake_pdf_ops)
    report_prompting = importlib.import_module(
        "backend.core.logic.report_analysis.report_prompting"
    )
    from backend.core.logic.report_analysis import analysis_cache

    analysis_cache.reset_cache()

    client = FakeAIClient()
    client.add_chat_response(
        '{"all_accounts": [{"name": "Cap One", "bureaus": ["Experian"]}], "inquiries": []}'
    )

    monkeypatch.setattr(
        report_prompting,
        "extract_account_headings",
        lambda text: [("cap one", "Cap One")],
    )
    monkeypatch.setattr(
        report_prompting,
        "extract_late_history_blocks",
        lambda text, return_raw_map=False: ({}, {}) if return_raw_map else {},
    )
    monkeypatch.setattr(report_prompting, "extract_inquiries", lambda text: [])
    monkeypatch.setattr(
        report_prompting, "normalize_creditor_name", lambda n: (n or "").lower()
    )

    out = tmp_path / "result.json"
    data = report_prompting.call_ai_analysis(
        "Experian section",
        False,
        out,
        ai_client=client,
        request_id="req",
        doc_fingerprint="fp_conf",
    )

    assert data["all_accounts"][0]["confidence"] == 1.0
    assert "needs_human_review" in data
    assert "missing_bureaus" in data


def test_call_ai_analysis_remediates_missing_account(tmp_path, monkeypatch):
    utils_pkg = types.ModuleType("backend.core.logic.utils")
    utils_pkg.__path__ = [
        str(
            Path(__file__).resolve().parents[1] / "backend" / "core" / "logic" / "utils"
        )
    ]
    sys.modules["backend.core.logic.utils"] = utils_pkg

    fake_pdf_ops = types.ModuleType("backend.core.logic.utils.pdf_ops")
    fake_pdf_ops.extract_pdf_text_safe = lambda *a, **k: ""
    sys.modules.setdefault("backend.core.logic.utils.pdf_ops", fake_pdf_ops)
    report_prompting = importlib.import_module(
        "backend.core.logic.report_analysis.report_prompting"
    )
    from backend.core.logic.report_analysis import analysis_cache

    analysis_cache.reset_cache()

    client = FakeAIClient()
    client.add_chat_response(
        '{"all_accounts": [{"name": "Cap One", "bureaus": ["Experian"]}, {"name": "Chase", "bureaus": ["Experian"]}, {"name": "Citi", "bureaus": ["Experian"]}], "inquiries": []}'
    )
    client.add_chat_response(
        '{"all_accounts": [{"name": "Cap One", "bureaus": ["Experian"]}, {"name": "Chase", "bureaus": ["Experian"]}, {"name": "Citi", "bureaus": ["Experian"]}, {"name": "BofA", "bureaus": ["Experian"]}], "inquiries": []}'
    )

    monkeypatch.setattr(
        report_prompting,
        "extract_account_headings",
        lambda text: [
            ("cap one", "Cap One"),
            ("chase", "Chase"),
            ("citi", "Citi"),
            ("bofa", "BofA"),
        ],
    )
    monkeypatch.setattr(
        report_prompting,
        "extract_late_history_blocks",
        lambda text, return_raw_map=False: ({}, {}) if return_raw_map else {},
    )
    monkeypatch.setattr(report_prompting, "extract_inquiries", lambda text: [])
    monkeypatch.setattr(
        report_prompting, "normalize_creditor_name", lambda n: (n or "").lower()
    )

    out = tmp_path / "result.json"
    data = report_prompting.call_ai_analysis(
        "Experian section",
        False,
        out,
        ai_client=client,
        request_id="req",
        doc_fingerprint="fp_rem_missing",
    )

    assert {a["name"] for a in data["all_accounts"]} == {
        "Cap One",
        "Chase",
        "Citi",
        "BofA",
    }
    assert len(client.chat_payloads) == 2
    assert "BofA" in client.chat_payloads[1]["messages"][0]["content"]


def test_call_ai_analysis_remediates_merged_account(tmp_path, monkeypatch):
    utils_pkg = types.ModuleType("backend.core.logic.utils")
    utils_pkg.__path__ = [
        str(
            Path(__file__).resolve().parents[1] / "backend" / "core" / "logic" / "utils"
        )
    ]
    sys.modules["backend.core.logic.utils"] = utils_pkg

    fake_pdf_ops = types.ModuleType("backend.core.logic.utils.pdf_ops")
    fake_pdf_ops.extract_pdf_text_safe = lambda *a, **k: ""
    sys.modules.setdefault("backend.core.logic.utils.pdf_ops", fake_pdf_ops)
    report_prompting = importlib.import_module(
        "backend.core.logic.report_analysis.report_prompting"
    )
    from backend.core.logic.report_analysis import analysis_cache

    analysis_cache.reset_cache()

    client = FakeAIClient()
    client.add_chat_response(
        '{"all_accounts": [{"name": "Cap One", "bureaus": ["Experian", "Equifax"]}], "inquiries": []}'
    )
    client.add_chat_response(
        '{"all_accounts": [{"name": "Cap One", "bureaus": ["Experian"]}], "inquiries": []}'
    )

    monkeypatch.setattr(
        report_prompting,
        "extract_account_headings",
        lambda text: [("cap one", "Cap One")],
    )
    monkeypatch.setattr(
        report_prompting,
        "extract_late_history_blocks",
        lambda text, return_raw_map=False: ({}, {}) if return_raw_map else {},
    )
    monkeypatch.setattr(report_prompting, "extract_inquiries", lambda text: [])
    monkeypatch.setattr(
        report_prompting, "normalize_creditor_name", lambda n: (n or "").lower()
    )

    out = tmp_path / "result.json"
    data = report_prompting.call_ai_analysis(
        "Experian section",
        False,
        out,
        ai_client=client,
        request_id="req",
        doc_fingerprint="fp_rem_merge",
    )

    assert data["all_accounts"][0]["bureaus"] == ["Experian"]
    assert len(client.chat_payloads) == 2
    assert "current bureau" in client.chat_payloads[1]["messages"][0]["content"].lower()


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

    from backend.core.logic.report_analysis import analysis_cache

    analysis_cache.reset_cache()

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
        request_id="req",
        doc_fingerprint="fp5",
    )

    result = analyze_report.analyze_credit_report(
        tmp_path / "dummy.pdf",
        tmp_path / "out.json",
        {"goal": "improve credit", "is_identity_theft": identity_theft},
        ai_client=client,
        request_id="req",
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
        request_id="req",
    )
    assert result["negative_accounts"] == []
    assert result["prompt_version"] == report_prompting.ANALYSIS_PROMPT_VERSION
    assert result["schema_version"] == report_prompting.ANALYSIS_SCHEMA_VERSION


def test_merge_accounts_under_merge():
    dest = [
        {
            "name": "ACME BANK",
            "opened_date": "01/01/2020",
            "balance": "$1,000",
            "bureaus": ["Experian"],
            "confidence": 0.5,
        }
    ]
    new = [
        {
            "name": "Acme Bank",
            "opened_date": "01/05/2020",
            "balance": "$1,005",
            "bureaus": ["Experian"],
            "confidence": 0.8,
        }
    ]
    report_prompting._merge_accounts(dest, new)
    assert len(dest) == 1
    assert dest[0]["confidence"] == 0.8


def test_merge_accounts_over_merge():
    dest = [
        {
            "name": "ACME BANK",
            "opened_date": "01/01/2020",
            "balance": "$1,000",
            "bureaus": ["Experian"],
        }
    ]
    new = [
        {
            "name": "Acme Bank",
            "opened_date": "03/15/2021",
            "balance": "$5,000",
            "bureaus": ["Experian"],
        }
    ]
    report_prompting._merge_accounts(dest, new)
    assert len(dest) == 2


def test_merge_inquiries_under_merge():
    dest = [{"creditor_name": "ACME BANK", "date": "01/01/2020", "bureau": "Experian"}]
    new = [{"creditor_name": "Acme Bank", "date": "01/07/2020", "bureau": "Experian"}]
    report_prompting._merge_inquiries(dest, new)
    assert len(dest) == 1


def test_merge_inquiries_over_merge():
    dest = [{"creditor_name": "ACME BANK", "date": "01/01/2020", "bureau": "Experian"}]
    new = [{"creditor_name": "Acme Bank", "date": "02/20/2020", "bureau": "Experian"}]
    report_prompting._merge_inquiries(dest, new)
    assert len(dest) == 2
