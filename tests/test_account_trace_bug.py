import logging
from pathlib import Path

import backend.core.orchestrators as orch
from backend.core.logic.report_analysis import analyze_report
from backend.core.logic.compliance import upload_validator
from backend.core.logic.report_analysis import report_postprocessing as rp
from backend.api import session_manager


def test_account_trace_bug(monkeypatch, caplog, tmp_path):
    monkeypatch.setenv("ANALYSIS_TRACE", "1")
    monkeypatch.setattr(orch, "get_ai_client", lambda: orch._StubAIClient())

    fake_sections = {
        "negative_accounts": [
            {
                "normalized_name": "Bad Bank",
                "name": "Bad Bank",
                "primary_issue": "charge_off",
                "issue_types": ["late"],
                "status": "open",
                "payment_statuses": {"Experian": "OK"},
                "payment_status": "OK",
                "has_co_marker": False,
                "remarks": "none",
                "source_stage": "analysis",
            }
        ],
        "open_accounts_with_issues": [],
        "all_accounts": [],
    }

    monkeypatch.setattr(
        analyze_report,
        "analyze_credit_report",
        lambda *args, **kwargs: fake_sections,
    )
    monkeypatch.setattr(upload_validator, "move_uploaded_file", lambda path, session_id: path)
    monkeypatch.setattr(upload_validator, "is_safe_pdf", lambda path: True)
    monkeypatch.setattr(rp, "_inject_missing_late_accounts", lambda *args, **kwargs: None)
    monkeypatch.setattr(rp, "enrich_account_metadata", lambda acc: acc)
    monkeypatch.setattr(session_manager, "update_session", lambda *args, **kwargs: None)

    pdf_file = tmp_path / "report.pdf"
    pdf_file.write_text("dummy")

    with caplog.at_level(logging.INFO):
        orch.extract_problematic_accounts_from_report(str(pdf_file), session_id="s1")
    bug_logs = [r for r in caplog.records if "account_trace_bug" in r.getMessage()]
    assert len(bug_logs) == 1


def test_account_trace_bug_skipped_with_details(monkeypatch, caplog, tmp_path):
    monkeypatch.setenv("ANALYSIS_TRACE", "1")
    monkeypatch.setattr(orch, "get_ai_client", lambda: orch._StubAIClient())

    fake_sections = {
        "negative_accounts": [
            {
                "normalized_name": "Bad Bank",
                "name": "Bad Bank",
                "primary_issue": "charge_off",
                "issue_types": ["late"],
                "status": "open",
                "payment_statuses": {"Experian": "OK"},
                "payment_status": "OK",
                "has_co_marker": False,
                "remarks": "none",
                "bureau_details": {
                    "Experian": {"account_status": "Charge-Off/Bad Debt"}
                },
                "source_stage": "analysis",
            }
        ],
        "open_accounts_with_issues": [],
        "all_accounts": [],
    }

    monkeypatch.setattr(
        analyze_report,
        "analyze_credit_report",
        lambda *args, **kwargs: fake_sections,
    )
    monkeypatch.setattr(upload_validator, "move_uploaded_file", lambda path, session_id: path)
    monkeypatch.setattr(upload_validator, "is_safe_pdf", lambda path: True)
    monkeypatch.setattr(rp, "_inject_missing_late_accounts", lambda *args, **kwargs: None)
    monkeypatch.setattr(rp, "enrich_account_metadata", lambda acc: acc)
    monkeypatch.setattr(session_manager, "update_session", lambda *args, **kwargs: None)

    pdf_file = tmp_path / "report.pdf"
    pdf_file.write_text("dummy")

    with caplog.at_level(logging.INFO):
        orch.extract_problematic_accounts_from_report(str(pdf_file), session_id="s1")
    bug_logs = [r for r in caplog.records if "account_trace_bug" in r.getMessage()]
    assert not bug_logs
