import time

import pytest

from backend.core.logic.report_analysis import analyze_report
from backend.core.telemetry import parser_metrics
from backend import config


@pytest.fixture(autouse=True)
def stub_parsing_dependencies(monkeypatch):
    """Stub heavy parsing helpers so tests can focus on telemetry."""

    monkeypatch.setattr(analyze_report, "extract_text_from_pdf", lambda _: "dummy")
    monkeypatch.setattr(analyze_report, "extract_account_headings", lambda _: [])
    monkeypatch.setattr(analyze_report, "_reconcile_account_headings", lambda *a, **k: None)
    monkeypatch.setattr(analyze_report, "extract_inquiries", lambda _: [])
    monkeypatch.setattr(analyze_report, "_merge_parser_inquiries", lambda *a, **k: None)
    monkeypatch.setattr(analyze_report, "extract_late_history_blocks", lambda *a, **k: ({}, {}, {}))
    monkeypatch.setattr(analyze_report, "_sanitize_late_counts", lambda *a, **k: None)
    monkeypatch.setattr(
        analyze_report, "extract_three_column_fields", lambda *a, **k: ({}, {}, {}, {}, {}, {}, {})
    )
    monkeypatch.setattr(analyze_report, "extract_payment_statuses", lambda *a, **k: ({}, {}))
    monkeypatch.setattr(analyze_report, "extract_creditor_remarks", lambda *a, **k: {})
    monkeypatch.setattr(analyze_report, "extract_account_numbers", lambda *a, **k: {})
    monkeypatch.setattr(analyze_report, "_inject_missing_late_accounts", lambda *a, **k: None)
    monkeypatch.setattr(analyze_report, "_cleanup_unverified_late_text", lambda *a, **k: None)
    monkeypatch.setattr(analyze_report, "_attach_parser_signals", lambda *a, **k: None)
    monkeypatch.setattr(analyze_report, "enrich_account_metadata", lambda acc: acc)
    monkeypatch.setattr(analyze_report, "validate_analysis_sanity", lambda _: [])


@pytest.fixture
def capture_events(monkeypatch):
    events = []

    def fake_emit(event, **fields):
        events.append((event, fields))

    monkeypatch.setattr(parser_metrics, "emit", fake_emit)
    return events


def _run(tmp_path, **kwargs):
    return analyze_report.analyze_credit_report(
        "dummy.pdf", tmp_path / "out.json", {}, request_id="rid", session_id="sid", **kwargs
    )


def test_all_text_pdf(tmp_path, monkeypatch, capture_events):
    monkeypatch.setattr(analyze_report, "extract_text_per_page", lambda _: ["x" * 70, "y" * 70])
    monkeypatch.setattr(config, "PDF_TEXT_MIN_CHARS_PER_PAGE", 64)

    _run(tmp_path, run_ai=False)

    event, payload = capture_events[0]
    assert event == "parser_audit"
    assert payload["pages_total"] == 2
    assert payload["pages_with_text"] == 2
    assert payload["pages_empty_text"] == 0
    assert payload["call_ai_ms"] is None
    assert payload["fields_written"] is None
    assert "x" * 70 not in str(payload)


def test_mixed_pages(tmp_path, monkeypatch, capture_events):
    pages = ["", "abc", " " * 10, "X" * 80]
    monkeypatch.setattr(analyze_report, "extract_text_per_page", lambda _: pages)
    monkeypatch.setattr(config, "PDF_TEXT_MIN_CHARS_PER_PAGE", 64)

    _run(tmp_path, run_ai=False)

    payload = capture_events[0][1]
    assert payload["pages_total"] == 4
    assert payload["pages_with_text"] == 1
    assert payload["pages_empty_text"] == 3


def test_ai_timing(tmp_path, monkeypatch, capture_events):
    monkeypatch.setattr(analyze_report, "extract_text_per_page", lambda _: ["x" * 70])

    def fake_ai(*a, **k):
        time.sleep(0.01)
        return {
            "negative_accounts": [],
            "open_accounts_with_issues": [],
            "positive_accounts": [],
            "high_utilization_accounts": [],
            "all_accounts": [],
            "inquiries": [],
            "needs_human_review": False,
            "missing_bureaus": [],
        }

    monkeypatch.setattr(analyze_report, "call_ai_analysis", fake_ai)

    _run(tmp_path, run_ai=True)

    assert capture_events[0][1]["call_ai_ms"] > 0


def test_error_path(tmp_path, monkeypatch, capture_events):
    def boom(_):
        raise RuntimeError("boom")

    monkeypatch.setattr(analyze_report, "extract_text_per_page", boom)

    _run(tmp_path, run_ai=False)

    payload = capture_events[0][1]
    assert payload["errors"] == "TextExtractionError"


def test_pii_safety(tmp_path, monkeypatch, capture_events):
    monkeypatch.setattr(analyze_report, "extract_text_per_page", lambda _: ["secret"])

    _run(tmp_path, run_ai=False)

    payload_str = str(capture_events[0][1])
    assert "secret" not in payload_str


def test_disabled_flag(tmp_path, monkeypatch, capture_events):
    monkeypatch.setattr(config, "PARSER_AUDIT_ENABLED", False)
    monkeypatch.setattr(analyze_report, "PARSER_AUDIT_ENABLED", False)
    monkeypatch.setattr(analyze_report, "extract_text_per_page", lambda _: ["x"])

    _run(tmp_path, run_ai=False)

    assert capture_events == []

