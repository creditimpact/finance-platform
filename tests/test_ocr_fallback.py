import pytest

from backend.core.logic.report_analysis import analyze_report, ocr_provider
from backend.core.telemetry import parser_metrics


@pytest.fixture(autouse=True)
def stub_parsing_dependencies(monkeypatch):
    """Stub heavy parsing helpers so tests can focus on OCR logic."""

    monkeypatch.setattr(analyze_report, "extract_account_headings", lambda _: [])
    monkeypatch.setattr(
        analyze_report, "_reconcile_account_headings", lambda *a, **k: None
    )
    monkeypatch.setattr(analyze_report, "extract_inquiries", lambda _: [])
    monkeypatch.setattr(analyze_report, "_merge_parser_inquiries", lambda *a, **k: None)
    monkeypatch.setattr(
        analyze_report, "extract_late_history_blocks", lambda *a, **k: ({}, {}, {})
    )
    monkeypatch.setattr(analyze_report, "_sanitize_late_counts", lambda *a, **k: None)
    monkeypatch.setattr(
        analyze_report,
        "extract_three_column_fields",
        lambda *a, **k: ({}, {}, {}, {}, {}, {}, {}),
    )
    monkeypatch.setattr(
        analyze_report, "extract_payment_statuses", lambda *a, **k: ({}, {})
    )
    monkeypatch.setattr(analyze_report, "extract_creditor_remarks", lambda *a, **k: {})
    monkeypatch.setattr(analyze_report, "extract_account_numbers", lambda *a, **k: {})
    monkeypatch.setattr(
        analyze_report, "_inject_missing_late_accounts", lambda *a, **k: None
    )
    monkeypatch.setattr(
        analyze_report, "_cleanup_unverified_late_text", lambda *a, **k: None
    )
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
        "dummy.pdf",
        tmp_path / "out.json",
        {},
        request_id="rid",
        session_id="sid",
        **kwargs,
    )


def test_no_ocr_when_disabled(tmp_path, monkeypatch, capture_events):
    monkeypatch.setattr(analyze_report, "OCR_ENABLED", False)
    monkeypatch.setattr(analyze_report, "extract_text_per_page", lambda _: ["page"])

    called = False

    def fake_get(name):  # pragma: no cover - provider should not be used
        nonlocal called
        called = True
        return None

    monkeypatch.setattr(analyze_report, "get_ocr_provider", fake_get)

    _run(tmp_path, run_ai=False)

    assert called is False
    payload = capture_events[0][1]
    assert payload["parser_pdf_pages_ocr"] == 0


def test_selective_ocr_merge(tmp_path, monkeypatch, capture_events):
    pages = ["", "X" * 10, "Y" * 80]
    monkeypatch.setattr(analyze_report, "extract_text_per_page", lambda _: pages.copy())
    monkeypatch.setattr(analyze_report, "OCR_ENABLED", True)
    monkeypatch.setattr(analyze_report, "PDF_TEXT_MIN_CHARS_PER_PAGE", 64)

    class Provider:
        def __init__(self):
            self.calls = []

        def ocr_page(self, pdf_path, page_index, *, timeout_ms, langs):
            self.calls.append(page_index)
            return ocr_provider.OcrResult(text=f"OCR{page_index}", duration_ms=5)

    provider = Provider()
    monkeypatch.setattr(analyze_report, "get_ocr_provider", lambda name: provider)

    captured_text = []

    def fake_extract_headings(txt):
        captured_text.append(txt)
        return []

    monkeypatch.setattr(
        analyze_report, "extract_account_headings", fake_extract_headings
    )

    _run(tmp_path, run_ai=False)

    assert provider.calls == [0, 1]
    text = captured_text[0]
    assert "OCR0" in text
    assert "OCR1" in text
    assert "Y" * 80 in text

    payload = capture_events[0][1]
    assert payload["parser_pdf_pages_ocr"] == 2
    assert payload["parser_ocr_latency_ms_total"] == 10
    assert payload["parser_ocr_errors"] == 0
    payload_str = str(payload)
    assert "OCR0" not in payload_str and "Y" * 80 not in payload_str


def test_ocr_timeout_counts_error(tmp_path, monkeypatch, capture_events):
    pages = ["", "Z" * 80]
    monkeypatch.setattr(analyze_report, "extract_text_per_page", lambda _: pages.copy())
    monkeypatch.setattr(analyze_report, "OCR_ENABLED", True)
    monkeypatch.setattr(analyze_report, "PDF_TEXT_MIN_CHARS_PER_PAGE", 64)

    class ErrProvider:
        def ocr_page(self, *a, **k):
            return ocr_provider.OcrResult(text="", duration_ms=7)

    monkeypatch.setattr(analyze_report, "get_ocr_provider", lambda name: ErrProvider())

    _run(tmp_path, run_ai=False)

    payload = capture_events[0][1]
    assert payload["parser_pdf_pages_ocr"] == 1
    assert payload["parser_ocr_errors"] == 1
