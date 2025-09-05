import logging
import pytest

from backend.core.logic.report_analysis import analyze_report
from backend.core.logic.report_analysis.extractors import accounts, sections, report_meta, summary
from backend.core.telemetry import metrics
from backend.core.case_store.models import SessionCase
from backend.core.case_store import api as cs_api
from backend.core.metrics import field_coverage
from backend.core.pdf import extract_text as pdf_extract


@pytest.fixture
def setup_env(monkeypatch):
    lines = [
        "JPMCB CARD",
        "Account # 426290**********",
        "Payment Status: Current",
        "Credit Limit: $18,900",
    ]
    text = "\n".join(lines)
    monkeypatch.setattr(
        analyze_report,
        "load_cached_text",
        lambda _sid: {"pages": [text], "full_text": text, "meta": {}},
    )
    monkeypatch.setattr(analyze_report, "char_count", lambda s: len(s))
    monkeypatch.setattr(analyze_report, "_reconcile_account_headings", lambda *a, **k: None)
    monkeypatch.setattr(analyze_report, "extract_inquiries", lambda t: [])
    monkeypatch.setattr(analyze_report, "extract_late_history_blocks", lambda *a, **k: ({}, {}, {}))
    monkeypatch.setattr(analyze_report, "_sanitize_late_counts", lambda *a, **k: None)
    monkeypatch.setattr(analyze_report, "extract_payment_statuses", lambda *a, **k: ({}, {}))
    monkeypatch.setattr(analyze_report, "extract_creditor_remarks", lambda *a, **k: {})
    monkeypatch.setattr(analyze_report, "extract_account_numbers", lambda *a, **k: {})
    monkeypatch.setattr(analyze_report, "_inject_missing_late_accounts", lambda *a, **k: None)
    monkeypatch.setattr(analyze_report, "_cleanup_unverified_late_text", lambda *a, **k: None)
    monkeypatch.setattr(analyze_report, "enrich_account_metadata", lambda acc: acc)
    monkeypatch.setattr(analyze_report, "validate_analysis_sanity", lambda _: [])
    monkeypatch.setattr(analyze_report, "extract_three_column_fields", lambda *a, **k: ({}, {}, {}, {}, {}, {}, {}))
    # Stub case store API
    stub_case = lambda sid: SessionCase(session_id=sid, accounts={})
    monkeypatch.setattr(cs_api, "load_session_case", stub_case)
    monkeypatch.setattr(cs_api, "save_session_case", lambda case: None)
    monkeypatch.setattr(cs_api, "create_session_case", lambda sid, meta=None: stub_case(sid))
    monkeypatch.setattr(cs_api, "list_accounts", lambda sid: [])
    monkeypatch.setattr(cs_api, "get_or_create_logical_account_id", lambda sid, lk: lk or "synthetic")
    monkeypatch.setattr(analyze_report, "load_session_case", stub_case)
    monkeypatch.setattr(analyze_report, "save_session_case", lambda case: None)
    monkeypatch.setattr(analyze_report, "create_session_case", lambda sid, meta=None: stub_case(sid))
    monkeypatch.setattr(report_meta, "load_session_case", stub_case)
    monkeypatch.setattr(report_meta, "save_session_case", lambda case: None)
    monkeypatch.setattr(accounts, "get_or_create_logical_account_id", lambda sid, lk: lk or "synthetic")
    monkeypatch.setattr(summary, "extract", lambda lines, session_id: {})
    monkeypatch.setattr(analyze_report, "attach_bureau_meta_tables", lambda *a, **k: None)
    monkeypatch.setattr(analyze_report, "_track_parse_pass", lambda *a, **k: None)
    monkeypatch.setattr(field_coverage, "emit_session_field_coverage_summary", lambda *a, **k: None)
    monkeypatch.setattr(pdf_extract, "extract_text", lambda *a, **k: "")
    monkeypatch.setattr(analyze_report, "OCR_ENABLED", False, raising=False)
    monkeypatch.setattr(analyze_report, "TEXT_NORMALIZE_ENABLED", False, raising=False)
    monkeypatch.setattr(analyze_report, "ENABLE_CASESTORE_WRITE", True, raising=False)
    monkeypatch.setattr(analyze_report, "DETERMINISTIC_EXTRACTORS_ENABLED", True, raising=False)
    monkeypatch.setattr(sections, "detect", lambda pages: {"bureaus": {"TransUnion": lines}, "report_meta": [], "summary": []})
    upserts = []
    monkeypatch.setattr(accounts, "upsert_account_fields", lambda **kw: upserts.append(kw))
    return lines, upserts


def run_analysis(tmp_path, session_id="sess", **kwargs):
    return analyze_report.analyze_credit_report(
        "dummy.pdf", tmp_path / "out.json", {}, request_id="rid", session_id=session_id, **kwargs
    )


def test_persist_when_header_detection_fails(tmp_path, setup_env, caplog):
    lines, upserts = setup_env
    caplog.set_level(logging.DEBUG, logger="backend.core.logic.report_analysis.analyze_report")
    run_analysis(tmp_path, session_id="sess1")
    assert upserts, "account should be persisted"
    fields = upserts[0]["fields"]
    assert fields.get("credit_limit") == 18900
    msg = "CASEBUILDER: three_column_header_missing"
    assert any(msg in rec.message for rec in caplog.records)


def test_no_drop_and_partial_by_bureau_ok(tmp_path, setup_env):
    lines, upserts = setup_env
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(
        analyze_report,
        "extract_payment_statuses",
        lambda text: ({"jpmcb card": {"TransUnion": "current"}}, {}),
    )
    res = run_analysis(tmp_path, session_id="sess2")
    assert upserts, "account should be persisted"
    assert res is not None
    monkeypatch.undo()


def test_metrics_recorded_but_not_counted_as_drop(tmp_path, setup_env, monkeypatch):
    lines, upserts = setup_env
    calls = []
    def fake_increment(name, value=1, tags=None):
        calls.append(name)
    monkeypatch.setattr(metrics, "increment", fake_increment)
    monkeypatch.setattr(
        analyze_report,
        "extract_payment_statuses",
        lambda text: ({"jpmcb card": {"TransUnion": "current"}}, {}),
    )
    run_analysis(tmp_path, session_id="sess3")
    assert "casebuilder.columns.header_missing" in calls
    assert "casebuilder.columns.fallback_used" in calls
    assert all("drop" not in c for c in calls)
    assert upserts, "account should still be persisted"
