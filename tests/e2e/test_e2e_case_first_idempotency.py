import copy
import importlib
import os
from pathlib import Path

import pytest

from backend.core.case_store import api as case_store, storage
from backend.core.metrics import field_coverage

from tests.helpers.case_asserts import (
    dict_superset,
    is_filled,
)

EXPECTED_FIELDS = {
    "EQ": field_coverage.EXPECTED_FIELDS["Equifax"],
    "TU": field_coverage.EXPECTED_FIELDS["TransUnion"],
    "EX": field_coverage.EXPECTED_FIELDS["Experian"],
}


def coverage_for_bureau(fields: dict, bureau: str) -> float:
    exp = EXPECTED_FIELDS[bureau]
    filled = sum(1 for k in exp if is_filled(fields.get(k)))
    return round(100 * filled / max(1, len(exp)))


def _make_fields() -> dict:
    base = {
        "balance_owed": 100.0,
        "payment_status": "current",
        "account_status": "open",
        "credit_limit": 1000.0,
        "past_due_amount": 0.0,
        "account_rating": "A",
        "account_description": "desc",
        "creditor_remarks": "none",
        "account_type": "revolving",
        "creditor_type": "bank",
        "dispute_status": "none",
        "two_year_payment_history": ["OK", "OK"],
        "days_late_7y": [],
        "normalized": {"status": "ok", "sources": ["eq", "tu", "ex"]},
    }
    by_bureau = {}
    for code in EXPECTED_FIELDS:
        bureau_fields = {f: f"{f}-{code}" for f in EXPECTED_FIELDS[code]}
        bureau_fields["payment_history"] = [
            {"date": "2023-01", "status": "OK"},
            {"date": "2023-02", "status": "OK"},
        ]
        by_bureau[code] = bureau_fields
    base["by_bureau"] = by_bureau
    return base


FAKE_REPORTS = {
    "golden_eq_tu_ex_basic.pdf": {"acc1": {"bureau": "Experian", "fields": _make_fields()}},
    "golden_eq_missing_limit.pdf": {"acc1": {"bureau": "Experian", "fields": _make_fields()}},
    "golden_tu_ex_history.pdf": {"acc1": {"bureau": "Experian", "fields": _make_fields()}},
}


def fake_analyze_credit_report(
    pdf_path,
    output_json_path,
    client_info,
    *,
    request_id,
    session_id,
    **kwargs,
):
    name = Path(pdf_path).name
    accounts = FAKE_REPORTS[name]
    try:
        case_store.load_session_case(session_id)
    except Exception:
        case_store.save_session_case(case_store.create_session_case(session_id))
    for acc_id, acc in accounts.items():
        case_store.upsert_account_fields(session_id, acc_id, acc["bureau"], acc["fields"])
    return {"session_id": session_id}


@pytest.mark.parametrize(
    "pdf_name",
    [
        "golden_eq_tu_ex_basic.pdf",
        "golden_eq_missing_limit.pdf",
        "golden_tu_ex_history.pdf",
    ],
)
def test_case_first_idempotency_no_data_loss(tmp_path, monkeypatch, pdf_name):
    monkeypatch.setenv("SAFE_MERGE_ENABLED", "1")
    monkeypatch.setenv("CASE_FIRST_BUILD_ENABLED", "1")
    monkeypatch.setenv("NORMALIZED_OVERLAY_ENABLED", "1")

    import backend.config as config
    import backend.core.config.flags as flags

    importlib.reload(flags)

    monkeypatch.setattr(config, "CASESTORE_DIR", str(tmp_path))
    monkeypatch.setattr(storage, "CASESTORE_DIR", tmp_path.as_posix())
    monkeypatch.setattr(config, "ENABLE_CASESTORE_WRITE", True)

    import backend.core.logic.report_analysis.analyze_report as analyze_report

    monkeypatch.setattr(analyze_report, "ENABLE_CASESTORE_WRITE", True, raising=False)
    monkeypatch.setattr(config, "ENABLE_CASESTORE_STAGEA", True)
    monkeypatch.setattr(config, "ENABLE_AI_ADJUDICATOR", False)
    monkeypatch.setattr(analyze_report, "analyze_credit_report", fake_analyze_credit_report)

    session_id = f"idemp-{pdf_name}"
    pdf_path = Path(__file__).parent / "fixtures/pdfs/golden" / pdf_name

    analyze_report.analyze_credit_report(
        str(pdf_path), tmp_path / "out.json", {}, request_id="rid", session_id=session_id
    )

    account_ids = case_store.list_accounts(session_id)
    for acc_id in account_ids:
        case = case_store.get_account_case(session_id, acc_id)
        by_bureau = getattr(case.fields, "by_bureau", {})
        for bureau, fields in by_bureau.items():
            cov = coverage_for_bureau(fields, bureau)
            assert cov >= 95
        if flags.NORMALIZED_OVERLAY_ENABLED:
            norm = getattr(case.fields, "normalized", None)
            assert norm and is_filled(norm.get("status")) and is_filled(norm.get("sources"))

    from backend.core.logic.report_analysis import problem_detection

    problem_detection.run_stage_a(session_id)
    for acc_id in account_ids:
        case = case_store.get_account_case(session_id, acc_id)
        artifact = case.artifacts.get("stageA_detection")
        assert artifact and is_filled(artifact.primary_issue) and is_filled(artifact.decision_source)

    before: dict[str, dict] = {}
    for acc_id in account_ids:
        case = case_store.get_account_case(session_id, acc_id)
        before[acc_id] = copy.deepcopy(case.fields.model_dump())

    analyze_report.analyze_credit_report(
        str(pdf_path), tmp_path / "out.json", {}, request_id="rid", session_id=session_id
    )

    for acc_id in account_ids:
        case = case_store.get_account_case(session_id, acc_id)
        after = copy.deepcopy(case.fields.model_dump())
        dict_superset(
            after,
            before[acc_id],
            {"payment_history": "date"},
        )

    problem_detection.run_stage_a(session_id)
    for acc_id in account_ids:
        case = case_store.get_account_case(session_id, acc_id)
        assert case.artifacts.get("stageA_detection")
