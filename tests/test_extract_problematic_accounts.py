import pytest

from backend.core.models import BureauPayload
from backend.core.orchestrators import (
    extract_problematic_accounts_from_report,
    extract_problematic_accounts_from_report_dict,
)


def _mock_dependencies(monkeypatch, sections):
    monkeypatch.setattr(
        "backend.core.logic.compliance.upload_validator.move_uploaded_file",
        lambda path, session_id: path,
    )
    monkeypatch.setattr(
        "backend.core.logic.compliance.upload_validator.is_safe_pdf", lambda path: True
    )
    monkeypatch.setattr(
        "backend.core.orchestrators.update_session", lambda *a, **k: None
    )
    monkeypatch.setattr(
        "backend.core.logic.report_analysis.analyze_report.analyze_credit_report",
        lambda *a, **k: sections,
    )


def test_extract_problematic_accounts_returns_models(monkeypatch):
    sections = {
        "negative_accounts": [{"name": "Acc1"}],
        "open_accounts_with_issues": [{"name": "Acc2"}],
        "unauthorized_inquiries": [{"creditor_name": "Bank", "date": "2024-01-01"}],
        "high_utilization_accounts": [{"name": "Acc3"}],
    }
    _mock_dependencies(monkeypatch, sections)
    payload = extract_problematic_accounts_from_report("dummy.pdf")
    assert isinstance(payload, BureauPayload)
    assert payload.disputes[0].name == "Acc1"
    assert payload.goodwill[0].name == "Acc2"
    assert payload.inquiries[0].creditor_name == "Bank"
    assert payload.high_utilization[0].name == "Acc3"


def test_extract_problematic_accounts_dict_adapter(monkeypatch):
    sections = {
        "negative_accounts": [{"name": "Acc1"}],
        "open_accounts_with_issues": [{"name": "Acc2"}],
        "unauthorized_inquiries": [{"creditor_name": "Bank", "date": "2024-01-01"}],
        "high_utilization_accounts": [{"name": "Acc3"}],
    }
    _mock_dependencies(monkeypatch, sections)
    with pytest.deprecated_call():
        result = extract_problematic_accounts_from_report_dict("dummy.pdf")
    assert isinstance(result, dict)
    assert result["negative_accounts"][0]["name"] == "Acc1"


def test_payload_to_dict(monkeypatch):
    sections = {
        "negative_accounts": [{"name": "Acc1"}],
        "open_accounts_with_issues": [{"name": "Acc2"}],
        "unauthorized_inquiries": [{"creditor_name": "Bank", "date": "2024-01-01"}],
        "high_utilization_accounts": [{"name": "Acc3"}],
    }
    _mock_dependencies(monkeypatch, sections)
    payload = extract_problematic_accounts_from_report("dummy.pdf")
    data = payload.to_dict()
    assert data["disputes"][0]["name"] == "Acc1"
    assert data["goodwill"][0]["name"] == "Acc2"
    assert data["inquiries"][0]["creditor_name"] == "Bank"
    assert data["high_utilization"][0]["name"] == "Acc3"


def test_parser_only_late_accounts_included(monkeypatch):
    from backend.core.logic.report_analysis.report_postprocessing import (
        _inject_missing_late_accounts,
    )

    result = {
        "all_accounts": [],
        "negative_accounts": [],
        "open_accounts_with_issues": [],
    }
    history = {"parser bank": {"Experian": {"30": 1}}}
    raw_map = {"parser bank": "Parser Bank"}
    _inject_missing_late_accounts(result, history, raw_map)
    for sec in ["negative_accounts", "open_accounts_with_issues"]:
        for acc in result.get(sec, []):
            for key in list(acc.keys()):
                if key != "name":
                    acc.pop(key)
    _mock_dependencies(monkeypatch, result)

    payload = extract_problematic_accounts_from_report("dummy.pdf")
    assert payload.disputes or payload.goodwill


def test_extract_problematic_accounts_without_openai(monkeypatch):
    from pathlib import Path
    from backend.core.logic.report_analysis.report_postprocessing import (
        _inject_missing_late_accounts,
    )

    def fake_analyze_report(pdf_path, analyzed_json_path, client_info, ai_client=None, run_ai=True, request_id=None):
        assert not run_ai
        result = {"all_accounts": [], "negative_accounts": [], "open_accounts_with_issues": []}
        history = {"parser bank": {"Experian": {"30": 1}}}
        raw_map = {"parser bank": "Parser Bank"}
        _inject_missing_late_accounts(result, history, raw_map)
        return result

    monkeypatch.setattr(
        "backend.core.logic.compliance.upload_validator.move_uploaded_file",
        lambda path, session_id: Path(path),
    )
    monkeypatch.setattr(
        "backend.core.logic.compliance.upload_validator.is_safe_pdf", lambda path: True
    )
    monkeypatch.setattr("backend.core.orchestrators.update_session", lambda *a, **k: None)
    monkeypatch.setattr(
        "backend.core.logic.report_analysis.analyze_report.analyze_credit_report",
        fake_analyze_report,
    )
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    payload = extract_problematic_accounts_from_report("dummy.pdf")
    assert payload.disputes and payload.disputes[0].name == "Parser Bank"
