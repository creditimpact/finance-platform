import logging
import pytest

from backend.config import CASESTORE_DIR, ENABLE_CASESTORE_WRITE  # noqa: F401
from backend import config
from backend.core.case_store import api as cs_api, storage
from backend.core.logic.report_analysis import analyze_report
import backend.core.pdf.extract_text as et


@pytest.fixture(autouse=True)
def stub_parsing(monkeypatch):
    monkeypatch.setattr(
        analyze_report,
        "load_cached_text",
        lambda _sid: {"pages": ["stub"], "full_text": "stub", "meta": {}},
    )
    monkeypatch.setattr(analyze_report, "char_count", lambda s: len(s))
    monkeypatch.setattr(analyze_report, "extract_account_blocks", lambda *_: [])
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
    monkeypatch.setattr(et, "extract_text", lambda *a, **k: "stub")


@pytest.fixture
def session(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CASESTORE_DIR", str(tmp_path))
    monkeypatch.setattr(storage, "CASESTORE_DIR", tmp_path.as_posix())
    monkeypatch.setattr(config, "ENABLE_CASESTORE_WRITE", True)
    monkeypatch.setattr(analyze_report, "ENABLE_CASESTORE_WRITE", True)
    session_id = "sess"
    case = cs_api.create_session_case(session_id)
    cs_api.save_session_case(case)
    return session_id


def _run(tmp_path, session_id):
    analyze_report.analyze_credit_report(
        "dummy.pdf", tmp_path / "out.json", {}, request_id="rid", session_id=session_id
    )


def test_single_parse_in_session(tmp_path, session):
    _run(tmp_path, session)
    case = cs_api.load_session_case(session)
    assert case.report_meta.raw_source.get("parse_passes") == 1


def test_multiple_invocations_still_counted(tmp_path, session, monkeypatch, caplog):
    metrics = []
    monkeypatch.setattr(analyze_report, "_emit_metric", lambda n, **t: metrics.append((n, t)))
    caplog.set_level(logging.WARNING)

    _run(tmp_path, session)
    _run(tmp_path, session)

    case = cs_api.load_session_case(session)
    assert case.report_meta.raw_source.get("parse_passes") == 2
    assert ("stage1.parse_multiple_passes", {"session_id": session}) in metrics
    assert any("stage1.parse_multiple_passes" in r.getMessage() for r in caplog.records)
