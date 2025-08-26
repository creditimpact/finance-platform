import importlib
import logging
from pathlib import Path

import pytest


@pytest.fixture
def mock_parser(monkeypatch, tmp_path):
    """Return a callable that runs the parser with deterministic fixtures."""

    def _run(caplog=None, level=logging.WARNING):
        # Reload config and related modules to pick up env vars set by tests
        import backend.config as config
        import backend.core.case_store.api as case_api
        import backend.core.logic.report_analysis.analyze_report as analyze_report

        importlib.reload(config)
        importlib.reload(case_api)
        importlib.reload(analyze_report)
        if caplog is not None:
            caplog.set_level(
                level, logger="backend.core.logic.report_analysis.analyze_report"
            )

        pdf_path = tmp_path / "report.pdf"
        pdf_path.write_text("dummy", encoding="utf-8")
        out_path = tmp_path / "out.json"

        session_id = "test-session"
        report_date = "2024-01-01"
        inquiries = [
            {"creditor_name": "Bank", "date": "2024-05-01", "bureau": "Equifax"}
        ]
        public_info = [{"type": "Bankruptcy"}]
        accounts = [
            {
                "name": "Account 1",
                "account_fingerprint": "acc1",
                "bureau_details": {
                    "Equifax": {
                        "account_number": "123456789",
                        "creditor_remarks": "john@example.com",
                        "account_status": "555-111-2222",
                        "account_description": "SSN 123-45-6789",
                        "balance_owed": 100,
                    }
                },
            },
            {
                "name": "Account 2",
                "account_fingerprint": "acc2",
                "bureau_details": {
                    "TransUnion": {
                        "account_number": "987654321",
                        "creditor_remarks": "no issues",
                        "account_status": "Open",
                        "credit_limit": 1000,
                        "balance_owed": 200,
                    }
                },
            },
        ]

        result = {
            "all_accounts": accounts,
            "negative_accounts": [],
            "open_accounts_with_issues": [],
            "positive_accounts": [],
            "high_utilization_accounts": [],
            "inquiries": inquiries,
            "public_information": public_info,
        }

        # Patch heavy helpers
        monkeypatch.setattr(analyze_report, "extract_text_from_pdf", lambda p: "text")
        monkeypatch.setattr(analyze_report, "extract_pdf_page_texts", lambda p: [])
        monkeypatch.setattr(analyze_report, "extract_account_headings", lambda t: [])
        monkeypatch.setattr(analyze_report, "extract_inquiries", lambda t: inquiries)
        monkeypatch.setattr(
            analyze_report, "extract_late_history_blocks", lambda *a, **k: ({}, {}, {})
        )
        monkeypatch.setattr(
            analyze_report, "extract_three_column_fields", lambda *a, **k: ({}, {}, {}, {}, {}, {}, {})
        )
        monkeypatch.setattr(
            analyze_report, "extract_payment_statuses", lambda *a, **k: ({}, {})
        )
        monkeypatch.setattr(analyze_report, "extract_creditor_remarks", lambda *a, **k: {})
        monkeypatch.setattr(analyze_report, "extract_account_numbers", lambda *a, **k: {})
        monkeypatch.setattr(analyze_report, "call_ai_analysis", lambda *a, **k: result)

        client_info = {"name": "John Doe", "report_date": report_date}

        output = analyze_report.analyze_credit_report(
            pdf_path,
            out_path,
            client_info,
            ai_client=None,
            run_ai=True,
            request_id="req",
            session_id=session_id,
        )
        return session_id, output, accounts, inquiries, public_info

    return _run


def test_flag_off(mock_parser, monkeypatch, tmp_path):
    monkeypatch.setenv("CASESTORE_DIR", str(tmp_path / "cases"))
    monkeypatch.setenv("ENABLE_CASESTORE_WRITE", "0")
    session_id, *_ = mock_parser()
    assert not (tmp_path / "cases" / f"{session_id}.json").exists()


def test_flag_on_no_redaction(mock_parser, monkeypatch, tmp_path):
    from backend.core.case_store.api import load_session_case

    monkeypatch.setenv("CASESTORE_DIR", str(tmp_path / "cases"))
    monkeypatch.setenv("ENABLE_CASESTORE_WRITE", "1")
    monkeypatch.setenv("CASESTORE_REDACT_BEFORE_STORE", "0")
    session_id, result, accounts, inquiries, public_info = mock_parser()

    case = load_session_case(session_id)
    assert str(case.report_meta.credit_report_date) == "2024-01-01"
    assert case.report_meta.personal_information.name == "John Doe"
    assert case.report_meta.inquiries[0]["creditor_name"] == inquiries[0]["creditor_name"]
    assert case.report_meta.inquiries[0]["date"] == inquiries[0]["date"]
    assert case.report_meta.public_information == public_info
    assert case.summary.total_accounts == 2
    eq_key = [k for k in case.accounts if k.endswith("_Equifax")][0]
    acc1 = case.accounts[eq_key].fields
    assert acc1.account_number == "123456789"
    assert acc1.creditor_remarks == "john@example.com"
    assert acc1.account_status == "555-111-2222"
    assert acc1.account_description == "SSN 123-45-6789"
    assert result["all_accounts"][0]["bureau_details"]["Equifax"]["account_number"] == "123456789"


def test_redaction_on(mock_parser, monkeypatch, tmp_path):
    from backend.core.case_store.api import load_session_case

    monkeypatch.setenv("CASESTORE_DIR", str(tmp_path / "cases"))
    monkeypatch.setenv("ENABLE_CASESTORE_WRITE", "1")
    monkeypatch.setenv("CASESTORE_REDACT_BEFORE_STORE", "1")
    session_id, result, *_ = mock_parser()

    case = load_session_case(session_id)
    eq_key = [k for k in case.accounts if k.endswith("_Equifax")][0]
    acc1 = case.accounts[eq_key].fields
    assert acc1.account_number == "****6789"
    assert acc1.creditor_remarks == "REDACTED_EMAIL"
    assert acc1.account_status == "REDACTED_PHONE"
    assert acc1.account_description == "REDACTED_SSN"
    assert acc1.balance_owed == 100
    tu_key = [k for k in case.accounts if k.endswith("_TransUnion")][0]
    acc2 = case.accounts[tu_key].fields
    assert acc2.account_number == "****4321"
    assert acc2.creditor_remarks == "no issues"
    assert acc2.account_status == "Open"
    assert acc2.credit_limit == 1000
    assert acc2.balance_owed == 200


def test_parity_logging(mock_parser, monkeypatch, tmp_path, caplog):
    monkeypatch.setenv("CASESTORE_DIR", str(tmp_path / "cases"))
    monkeypatch.setenv("ENABLE_CASESTORE_WRITE", "1")
    monkeypatch.setenv("CASESTORE_REDACT_BEFORE_STORE", "1")
    monkeypatch.setenv("CASESTORE_PARSER_LOG_PARITY", "1")
    mock_parser(caplog, level=logging.INFO)
    lines = [r for r in caplog.records if "casestore_parity" in r.getMessage()]
    assert len(lines) == 2
    for rec in lines:
        msg = rec.getMessage()
        assert "added=" in msg and "changed=" in msg and "masked=" in msg and "missing=" in msg


def test_io_failure(mock_parser, monkeypatch, tmp_path, caplog):
    monkeypatch.setenv("CASESTORE_DIR", "/dev/null")
    monkeypatch.setenv("ENABLE_CASESTORE_WRITE", "1")
    monkeypatch.setenv("CASESTORE_REDACT_BEFORE_STORE", "0")
    session_id, result, accounts, *_ = mock_parser(caplog)
    assert result["all_accounts"] == accounts
    # ensure no case file was created and run did not crash
    assert not Path("/dev/null").joinpath(f"{session_id}.json").exists()
