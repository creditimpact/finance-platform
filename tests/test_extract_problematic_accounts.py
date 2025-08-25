import copy
import json
import logging
from hashlib import sha1

import pytest

from backend.core.logic.utils.names_normalization import normalize_creditor_name
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
        "negative_accounts": [{"name": "Acc1", "issue_types": ["late_payment"]}],
        "open_accounts_with_issues": [
            {"name": "Acc2", "issue_types": ["late_payment"]}
        ],
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
        "negative_accounts": [{"name": "Acc1", "issue_types": ["late_payment"]}],
        "open_accounts_with_issues": [
            {"name": "Acc2", "issue_types": ["late_payment"]}
        ],
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
        "negative_accounts": [{"name": "Acc1", "issue_types": ["late_payment"]}],
        "open_accounts_with_issues": [
            {"name": "Acc2", "issue_types": ["late_payment"]}
        ],
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


def test_backfills_default_fields(monkeypatch):
    sections = {
        "negative_accounts": [{"name": "Acc1", "issue_types": ["late_payment"]}],
        "open_accounts_with_issues": [
            {"name": "Acc2", "issue_types": ["late_payment"]}
        ],
        "unauthorized_inquiries": [],
        "high_utilization_accounts": [],
    }
    _mock_dependencies(monkeypatch, sections)
    monkeypatch.setattr(
        "backend.core.logic.report_analysis.report_postprocessing.enrich_account_metadata",
        lambda acc: acc,
    )
    payload = extract_problematic_accounts_from_report("dummy.pdf")
    expected = {
        "primary_issue": "unknown",
        "issue_types": ["late_payment"],
        "status": "",
        "late_payments": {},
        "payment_statuses": {},
        "has_co_marker": False,
        "co_bureaus": [],
        "remarks_contains_co": False,
        "bureau_statuses": {},
        "account_number_last4": None,
        "account_fingerprint": None,
        "original_creditor": None,
        "source_stage": "",
        "bureau_details": {},
    }
    neg = payload.disputes[0].to_dict()
    good = payload.goodwill[0].to_dict()
    for acc in (neg, good):
        for key, val in expected.items():
            assert acc.get(key) == val


def test_high_utilization_backfills_default_fields(monkeypatch):
    sections = {
        "negative_accounts": [],
        "open_accounts_with_issues": [],
        "unauthorized_inquiries": [],
        "high_utilization_accounts": [{"name": "Acc3"}],
    }
    _mock_dependencies(monkeypatch, sections)
    monkeypatch.setattr(
        "backend.core.logic.report_analysis.report_postprocessing.enrich_account_metadata",
        lambda acc: acc,
    )
    payload = extract_problematic_accounts_from_report("dummy.pdf")
    expected = {
        "primary_issue": "unknown",
        "issue_types": [],
        "status": "",
        "late_payments": {},
        "payment_statuses": {},
        "has_co_marker": False,
        "co_bureaus": [],
        "remarks_contains_co": False,
        "bureau_statuses": {},
        "account_number_last4": None,
        "account_fingerprint": None,
        "original_creditor": None,
        "source_stage": "",
        "bureau_details": {},
    }
    high = payload.high_utilization[0].to_dict()
    for key, val in expected.items():
        assert high.get(key) == val


def test_parser_only_late_accounts_excluded_when_flag_set(monkeypatch):
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
    _inject_missing_late_accounts(result, history, raw_map, {})
    for sec in ["negative_accounts", "open_accounts_with_issues"]:
        for acc in result.get(sec, []):
            for key in list(acc.keys()):
                if key not in {"name", "issue_types"}:
                    acc.pop(key)
    _mock_dependencies(monkeypatch, result)
    monkeypatch.setattr(
        "backend.core.orchestrators.EXCLUDE_PARSER_AGGREGATED_ACCOUNTS", True
    )

    payload = extract_problematic_accounts_from_report("dummy.pdf")
    assert not payload.disputes and not payload.goodwill


def test_extract_problematic_accounts_without_openai(monkeypatch):
    from pathlib import Path

    from backend.core.logic.report_analysis.report_postprocessing import (
        _inject_missing_late_accounts,
    )

    def fake_analyze_report(
        pdf_path,
        analyzed_json_path,
        client_info,
        ai_client=None,
        run_ai=True,
        request_id=None,
    ):
        assert not run_ai
        result = {
            "all_accounts": [],
            "negative_accounts": [],
            "open_accounts_with_issues": [],
        }
        history = {"parser bank": {"Experian": {"30": 1}}}
        raw_map = {"parser bank": "Parser Bank"}
        _inject_missing_late_accounts(result, history, raw_map, {})
        return result

    monkeypatch.setattr(
        "backend.core.logic.compliance.upload_validator.move_uploaded_file",
        lambda path, session_id: Path(path),
    )
    monkeypatch.setattr(
        "backend.core.logic.compliance.upload_validator.is_safe_pdf", lambda path: True
    )
    monkeypatch.setattr(
        "backend.core.orchestrators.update_session", lambda *a, **k: None
    )
    monkeypatch.setattr(
        "backend.core.logic.report_analysis.analyze_report.analyze_credit_report",
        fake_analyze_report,
    )
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    payload = extract_problematic_accounts_from_report("dummy.pdf")
    assert [a.name for a in payload.disputes] == ["Parser Bank"]
    assert not payload.goodwill


def test_extract_problematic_accounts_filters_out_clean_accounts(monkeypatch):
    sections = {
        "negative_accounts": [
            {"name": "Bad", "issue_types": ["late_payment"]},
            {"name": "Clean"},
        ],
        "open_accounts_with_issues": [
            {"name": "GoodwillBad", "issue_types": ["late_payment"]},
            {"name": "GoodwillClean"},
        ],
    }
    _mock_dependencies(monkeypatch, sections)
    payload = extract_problematic_accounts_from_report("dummy.pdf")
    assert [a.name for a in payload.disputes] == ["Bad"]
    assert [a.name for a in payload.goodwill] == ["GoodwillBad"]


def test_enriched_metadata_present_ai(monkeypatch):
    sections = {
        "negative_accounts": [
            {
                "name": "Acme Bank",
                "account_number": "1234567890",
                "original_creditor": "ACME",
                "account_type": "Credit Card",
                "bureaus": [
                    {
                        "bureau": "Experian",
                        "status": "Collection",
                        "balance": 1000,
                        "past_due": 100,
                        "date_opened": "2020-01-01",
                        "date_closed": "2022-01-01",
                        "last_activity": "2023-12-01",
                    },
                    {"bureau": "TransUnion", "status": "Open"},
                ],
                "issue_types": ["collection"],
            }
        ],
        "open_accounts_with_issues": [],
        "unauthorized_inquiries": [],
        "high_utilization_accounts": [],
    }
    _mock_dependencies(monkeypatch, sections)

    def fake_analyze_report(
        pdf_path,
        analyzed_json_path,
        client_info,
        ai_client=None,
        run_ai=True,
        request_id=None,
    ):
        assert run_ai
        return sections

    monkeypatch.setattr(
        "backend.core.logic.report_analysis.analyze_report.analyze_credit_report",
        fake_analyze_report,
    )
    monkeypatch.setattr("backend.core.orchestrators.get_ai_client", lambda: object())
    monkeypatch.delenv("ANALYSIS_FORCE_PARSER_ONLY", raising=False)

    payload = extract_problematic_accounts_from_report("dummy.pdf")
    acc = payload.to_dict()["disputes"][0]
    assert acc["normalized_name"] == normalize_creditor_name("Acme Bank")
    assert acc["account_number_last4"] == "7890"
    assert acc["original_creditor"] == "ACME"
    assert acc["account_type"] == "Credit Card"
    assert acc["balance"] == 1000
    assert acc["past_due"] == 100
    assert acc["date_opened"] == "2020-01-01"
    assert acc["date_closed"] == "2022-01-01"
    assert acc["last_activity"] == "2023-12-01"
    assert acc["bureau_statuses"] == {
        "Experian": "Collection/Chargeoff",
        "TransUnion": "Open/Current",
    }
    assert acc["source_stage"] == "ai_final"


def test_account_fingerprint_added_when_last4_missing(monkeypatch):
    sections = {
        "negative_accounts": [
            {
                "name": "Acme Bank",
                "date_opened": "2020-01-01",
                "late_payments": {"30": 1},
                "issue_types": ["late_payment"],
            }
        ],
        "open_accounts_with_issues": [],
        "unauthorized_inquiries": [],
        "high_utilization_accounts": [],
    }
    _mock_dependencies(monkeypatch, sections)
    payload = extract_problematic_accounts_from_report("dummy.pdf")
    acc = payload.to_dict()["disputes"][0]
    assert acc["account_number_last4"] is None
    expected = sha1(
        f"{normalize_creditor_name('Acme Bank')}|2020-01-01|30".encode()
    ).hexdigest()[:8]
    assert acc["account_fingerprint"] == expected


def test_enriched_metadata_present_parser_only(monkeypatch):
    sections = {
        "negative_accounts": [
            {
                "name": "Acme Bank",
                "account_number": "1234567890",
                "original_creditor": "ACME",
                "account_type": "Credit Card",
                "bureaus": [
                    {
                        "bureau": "Experian",
                        "status": "Collection",
                        "balance": 1000,
                        "past_due": 100,
                        "date_opened": "2020-01-01",
                        "date_closed": "2022-01-01",
                        "last_activity": "2023-12-01",
                    },
                    {"bureau": "TransUnion", "status": "Open"},
                ],
                "issue_types": ["collection"],
            }
        ],
        "open_accounts_with_issues": [],
        "unauthorized_inquiries": [],
        "high_utilization_accounts": [],
    }
    _mock_dependencies(monkeypatch, sections)

    def fake_analyze_report(
        pdf_path,
        analyzed_json_path,
        client_info,
        ai_client=None,
        run_ai=True,
        request_id=None,
    ):
        assert not run_ai
        return sections

    monkeypatch.setattr(
        "backend.core.logic.report_analysis.analyze_report.analyze_credit_report",
        fake_analyze_report,
    )
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    payload = extract_problematic_accounts_from_report("dummy.pdf")
    acc = payload.to_dict()["disputes"][0]
    assert acc["normalized_name"] == normalize_creditor_name("Acme Bank")
    assert acc["account_number_last4"] == "7890"
    assert acc["original_creditor"] == "ACME"
    assert acc["account_type"] == "Credit Card"
    assert acc["balance"] == 1000
    assert acc["past_due"] == 100
    assert acc["date_opened"] == "2020-01-01"
    assert acc["date_closed"] == "2022-01-01"
    assert acc["last_activity"] == "2023-12-01"
    assert acc["bureau_statuses"] == {
        "Experian": "Collection/Chargeoff",
        "TransUnion": "Open/Current",
    }
    assert acc["source_stage"] == "ai_final"


def test_tri_merge_toggle_does_not_change_counts_or_primary_issue(monkeypatch, caplog):
    base_sections = {
        "negative_accounts": [
            {
                "account_id": "1",
                "name": "Cap One",
                "account_number": "1234",
                "bureaus": ["Experian"],
                "issue_types": ["late_payment"],
                "primary_issue": "late_payment",
            }
        ],
        "open_accounts_with_issues": [
            {
                "account_id": "2",
                "name": "Cap One",
                "account_number": "1234",
                "bureaus": ["Equifax"],
                "issue_types": ["late_payment"],
                "primary_issue": "late_payment",
            }
        ],
    }

    # Tri-merge disabled baseline
    monkeypatch.delenv("ENABLE_TRI_MERGE", raising=False)
    _mock_dependencies(monkeypatch, copy.deepcopy(base_sections))
    with caplog.at_level(logging.INFO, logger="backend.core.orchestrators"):
        payload = extract_problematic_accounts_from_report("dummy.pdf")
    neg_before = len(payload.disputes)
    open_before = len(payload.goodwill)
    baseline_trace = [
        json.loads(r.getMessage().split("account_trace ")[1])
        for r in caplog.records
        if "account_trace" in r.getMessage()
    ]
    caplog.clear()

    # Tri-merge enabled
    _mock_dependencies(monkeypatch, copy.deepcopy(base_sections))
    monkeypatch.setenv("ENABLE_TRI_MERGE", "1")
    with caplog.at_level(logging.INFO, logger="backend.core.orchestrators"):
        payload_tri = extract_problematic_accounts_from_report("dummy.pdf")

    assert len(payload_tri.disputes) == neg_before
    assert len(payload_tri.goodwill) == open_before
    tri_trace = [
        json.loads(r.getMessage().split("account_trace ")[1])
        for r in caplog.records
        if "account_trace" in r.getMessage()
    ]
    assert [t["primary_issue"] for t in tri_trace] == [
        t["primary_issue"] for t in baseline_trace
    ]
    assert len(tri_trace) == len(baseline_trace)


def test_extract_problematic_accounts_logs_enriched_metadata(monkeypatch, caplog):
    sections = {
        "negative_accounts": [
            {
                "name": "Acc1",
                "issue_types": ["late_payment"],
                "payment_status": "current",
                "has_co_marker": False,
                "remarks": "note",
            }
        ]
    }
    _mock_dependencies(monkeypatch, sections)
    with caplog.at_level(logging.INFO, logger="backend.core.orchestrators"):
        payload = extract_problematic_accounts_from_report("dummy.pdf")
    assert isinstance(payload, BureauPayload)
    assert any(
        r.message.startswith("emitted_account name=acc1")
        and "primary_issue=" in r.message
        and "status=" in r.message
        and "last4=" in r.message
        and "orig_cred=" in r.message
        and "issues=" in r.message
        and "bureaus=" in r.message
        and "stage=" in r.message
        and "payment_statuses=current" in r.message
        and "has_co_marker=False" in r.message
        and "has_remarks=True" in r.message
        for r in caplog.records
    )


def test_ai_failure_falls_back_to_parser(monkeypatch):
    from pathlib import Path

    def fake_analyze_report(
        pdf_path,
        analyzed_json_path,
        client_info,
        ai_client=None,
        run_ai=True,
        request_id=None,
    ):
        if run_ai:
            return {
                "ai_failed": True,
                "all_accounts": [],
                "negative_accounts": [],
                "open_accounts_with_issues": [],
            }
        history = {"parser bank": {"Experian": {"30": 1}}}
        raw_map = {"parser bank": "Parser Bank"}
        from backend.core.logic.report_analysis.report_postprocessing import (
            _inject_missing_late_accounts,
        )

        result = {
            "all_accounts": [],
            "negative_accounts": [],
            "open_accounts_with_issues": [],
        }
        _inject_missing_late_accounts(result, history, raw_map, {})
        result["needs_human_review"] = True
        return result

    monkeypatch.setattr(
        "backend.core.logic.compliance.upload_validator.move_uploaded_file",
        lambda path, session_id: Path(path),
    )
    monkeypatch.setattr(
        "backend.core.logic.compliance.upload_validator.is_safe_pdf", lambda path: True
    )
    monkeypatch.setattr(
        "backend.core.orchestrators.update_session", lambda *a, **k: None
    )
    monkeypatch.setattr(
        "backend.core.logic.report_analysis.analyze_report.analyze_credit_report",
        fake_analyze_report,
    )
    monkeypatch.setattr("backend.core.orchestrators.get_ai_client", lambda: object())
    monkeypatch.delenv("ANALYSIS_FORCE_PARSER_ONLY", raising=False)

    payload = extract_problematic_accounts_from_report("dummy.pdf")
    assert [a.name for a in payload.disputes] == ["Parser Bank"]
    assert getattr(payload, "needs_human_review", False)
