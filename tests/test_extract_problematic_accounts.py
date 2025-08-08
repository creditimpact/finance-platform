import pytest
from orchestrators import (
    extract_problematic_accounts_from_report,
    extract_problematic_accounts_from_report_dict,
)
from models import BureauPayload


def _mock_dependencies(monkeypatch, sections):
    monkeypatch.setattr(
        "logic.upload_validator.move_uploaded_file", lambda path, session_id: path
    )
    monkeypatch.setattr("logic.upload_validator.is_safe_pdf", lambda path: True)
    monkeypatch.setattr("session_manager.update_session", lambda *a, **k: None)
    monkeypatch.setattr(
        "logic.analyze_report.analyze_credit_report", lambda *a, **k: sections
    )


def test_extract_problematic_accounts_returns_models(monkeypatch):
    sections = {
        "negative_accounts": [{"name": "Acc1"}],
        "open_accounts_with_issues": [{"name": "Acc2"}],
        "unauthorized_inquiries": [
            {"creditor_name": "Bank", "date": "2024-01-01"}
        ],
    }
    _mock_dependencies(monkeypatch, sections)
    payload = extract_problematic_accounts_from_report("dummy.pdf")
    assert isinstance(payload, BureauPayload)
    assert payload.disputes[0].name == "Acc1"
    assert payload.goodwill[0].name == "Acc2"
    assert payload.inquiries[0].creditor_name == "Bank"


def test_extract_problematic_accounts_dict_adapter(monkeypatch):
    sections = {
        "negative_accounts": [{"name": "Acc1"}],
        "open_accounts_with_issues": [{"name": "Acc2"}],
        "unauthorized_inquiries": [
            {"creditor_name": "Bank", "date": "2024-01-01"}
        ],
    }
    _mock_dependencies(monkeypatch, sections)
    with pytest.deprecated_call():
        result = extract_problematic_accounts_from_report_dict("dummy.pdf")
    assert isinstance(result, dict)
    assert result["negative_accounts"][0]["name"] == "Acc1"
