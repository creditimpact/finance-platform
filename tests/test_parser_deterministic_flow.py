import json
import warnings
import pytest
from pathlib import Path

from backend.config import (
    DETERMINISTIC_EXTRACTORS_ENABLED,
    ENABLE_CASESTORE_WRITE,
    TEXT_NORMALIZE_ENABLED,
)
from backend.core.case_store import api, storage
from backend.core.logic.report_analysis import analyze_report as ar
from backend.core.telemetry import parser_metrics

warnings.filterwarnings("ignore", category=DeprecationWarning)


def setup_env(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "CASESTORE_DIR", tmp_path.as_posix())
    monkeypatch.setattr(ar, "extract_text_per_page", lambda _: SAMPLE_PAGES)
    monkeypatch.setattr(ar, "extract_text_from_pdf", lambda _: "\n".join(SAMPLE_PAGES))
    monkeypatch.setattr(ar, "extract_account_headings", lambda text: [])
    monkeypatch.setattr(ar, "extract_inquiries", lambda text: [])
    monkeypatch.setattr(ar, "extract_late_history_blocks", lambda *args, **kwargs: ({}, {}, {}))
    monkeypatch.setattr(
        ar, "extract_three_column_fields", lambda *args, **kwargs: ({}, {}, {}, {}, {}, {}, {})
    )
    monkeypatch.setattr(ar, "extract_creditor_remarks", lambda text: {})
    monkeypatch.setattr(ar, "extract_payment_statuses", lambda text: ({}, {}))
    monkeypatch.setattr(ar, "_sanitize_late_counts", lambda x: None)
    monkeypatch.setattr(ar, "_normalize_keys", lambda x: x)
    monkeypatch.setattr(parser_metrics, "emit_parser_audit", lambda **kw: TELEMETRY.append(kw))
    monkeypatch.setattr(ar, "emit_parser_audit", lambda **kw: TELEMETRY.append(kw))
    monkeypatch.setattr(ar, "TEXT_NORMALIZE_ENABLED", False, raising=False)
    monkeypatch.setattr(ar, "DETERMINISTIC_EXTRACTORS_ENABLED", True, raising=False)
    monkeypatch.setattr(ar, "ENABLE_CASESTORE_WRITE", True, raising=False)


SAMPLE_PAGES = [
    "Experian Accounts\nAccount # 1111\nBalance Owed: $100\n\nEquifax Accounts\nAccount # 2222\nBalance Owed: $200\n\nReport Meta\nName: Jane Doe\nCredit Report Date: 2024-05-01\n\nSummary\nTotal Accounts: 2\nOpen Accounts: 2\nClosed Accounts: 0\n",
]
TELEMETRY = []


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_deterministic_flow(tmp_path, monkeypatch):
    setup_env(tmp_path, monkeypatch)
    output = tmp_path / "out.json"
    case = api.create_session_case("sess")
    api.save_session_case(case)
    ar.analyze_credit_report(
        pdf_path="dummy.pdf",
        client_info={},
        output_json_path=str(output),
        session_id="sess",
        request_id="req1",
    )
    stored = api.load_session_case("sess")
    assert len(stored.accounts) == 2
    assert stored.summary.total_accounts == 2
    assert str(stored.report_meta.credit_report_date) == "2024-05-01"
    assert TELEMETRY and TELEMETRY[0]["extractor_accounts_total"]["Experian"] == 1
