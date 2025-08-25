import pytest

import backend.core.logic.report_analysis.report_postprocessing as rp


def test_assign_issue_types_detects_late_payment():
    acc = {"late_payments": {"30": 1}}
    rp._assign_issue_types(acc)
    assert acc["issue_types"] == ["late_payment"]
    assert acc["primary_issue"] == "late_payment"
    assert acc["status"] == "Delinquent"
    assert acc["advisor_comment"] == "Late payments detected"


def test_assign_issue_types_collection_from_status():
    acc = {"status": "Sent to collections"}
    rp._assign_issue_types(acc)
    assert acc["issue_types"] == ["collection"]
    assert acc["primary_issue"] == "collection"
    assert acc["status"] == "Collection"


def test_assign_issue_types_charge_off_from_flags():
    acc = {"flags": ["Charge-Off"]}
    rp._assign_issue_types(acc)
    assert acc["issue_types"] == ["charge_off"]
    assert acc["primary_issue"] == "charge_off"
    assert acc["status"] == "Charge Off"


def test_assign_issue_types_collection_overrides_late():
    acc = {"status": "Sent to collections", "late_payments": {"30": 2}}
    rp._assign_issue_types(acc)
    assert acc["issue_types"] == ["collection", "late_payment"]
    assert acc["primary_issue"] == "collection"
    assert acc["status"] == "Collection"
    assert acc["advisor_comment"] == "Account in collection"


def test_assign_issue_types_collection_from_payment_status_only():
    acc = {"payment_status": "Account sent to collection agency"}
    rp._assign_issue_types(acc)
    assert acc["issue_types"] == ["collection"]
    assert acc["primary_issue"] == "collection"
    assert acc["status"] == "Collection"


def test_assign_issue_types_charge_off_overrides_late():
    acc = {"flags": ["Charge-Off"], "late_payments": {"Experian": {"30": 1}}}
    rp._assign_issue_types(acc)
    assert acc["issue_types"] == ["charge_off", "late_payment"]
    assert acc["primary_issue"] == "charge_off"
    assert acc["status"] == "Charge Off"
    assert acc["advisor_comment"] == "Account charged off"


def test_assign_issue_types_detects_charge_off_from_history_grid():
    acc = {"history": "OK OK CO"}
    rp._assign_issue_types(acc)
    assert acc["has_co_marker"] is True
    assert acc["issue_types"] == ["charge_off"]
    assert acc["primary_issue"] == "charge_off"
    assert acc["status"] == "Charge Off"


def test_assign_issue_types_detects_charge_off_from_late_map():
    acc = {"late_payments": {"Experian": {"CO": 1}}}
    rp._assign_issue_types(acc)
    assert acc["has_co_marker"] is True
    assert acc["issue_types"] == ["charge_off", "late_payment"]
    assert acc["primary_issue"] == "charge_off"
    assert acc["status"] == "Charge Off"


def test_assign_issue_types_co_grid_with_late_counts():
    acc = {
        "late_payments": {"Experian": {"30": 1}},
        "late_payment_history": {"Experian": "OK OK CO"},
    }
    rp._assign_issue_types(acc)
    assert acc["has_co_marker"] is True
    assert acc["issue_types"] == ["charge_off", "late_payment"]
    assert acc["primary_issue"] == "charge_off"
    assert acc["status"] == "Charge Off"
    assert acc.get("co_bureaus") == ["Experian"]


@pytest.mark.parametrize(
    "text",
    [
        "Account placed in collection",
        "Account transferred to collection agency",
    ],
)
def test_assign_issue_types_collection_from_remarks(text):
    acc = {"remarks": text}
    rp._assign_issue_types(acc)
    assert acc["has_co_marker"] is True
    assert acc["remarks_contains_co"] is True
    assert acc["issue_types"] == ["collection"]
    assert acc["primary_issue"] == "collection"
    assert acc["status"] == "Collection"


def test_assign_issue_types_collection_from_payment_status_overrides_late():
    acc = {"payment_status": "Sent to collection agency", "late_payments": {"30": 1}}
    rp._assign_issue_types(acc)
    assert acc["issue_types"] == ["collection", "late_payment"]
    assert acc["primary_issue"] == "collection"
    assert acc["status"] == "Collection"
    assert acc["advisor_comment"] == "Account in collection"


def test_assign_issue_types_charge_off_from_remarks_overrides_late():
    acc = {
        "remarks": "Account charged off as bad debt",
        "late_payments": {"Experian": {"30": 2}},
    }
    rp._assign_issue_types(acc)
    assert acc["issue_types"] == ["charge_off", "late_payment"]
    assert acc["primary_issue"] == "charge_off"
    assert acc["status"] == "Charge Off"
    assert acc["advisor_comment"] == "Account charged off"


def test_assign_issue_types_from_payment_statuses_map():
    acc = {
        "payment_statuses": {"TransUnion": "Collection/Chargeoff"},
        "late_payments": {"TransUnion": {"30": 1}},
    }
    rp._assign_issue_types(acc)
    assert acc["primary_issue"] == "collection"
    assert acc["issue_types"] == ["collection", "charge_off", "late_payment"]


def test_enrich_account_metadata_sets_last4_from_bureaus():
    acc = {
        "name": "Acme Bank",
        "bureaus": [{"bureau": "Experian", "account_number": "123-456-7890"}],
    }
    rp.enrich_account_metadata(acc)
    assert acc["account_number_last4"] == "7890"
    assert "account_fingerprint" not in acc


def test_enrich_account_metadata_uses_account_number_raw():
    acc = {
        "name": "Acme Bank",
        "bureaus": [{"bureau": "TransUnion", "account_number_raw": "****4321"}],
    }
    rp.enrich_account_metadata(acc)
    assert acc["account_number_last4"] == "4321"
    assert "account_fingerprint" not in acc


def test_enrich_account_metadata_skips_when_no_digits():
    acc = {
        "name": "Acme Bank",
        "account_number_raw": "t disputed",
        "bureaus": [{"bureau": "Experian", "account_number_raw": "N/A"}],
    }
    rp.enrich_account_metadata(acc)
    assert "account_number_last4" not in acc
    assert "account_number_raw" not in acc
    assert all("account_number_raw" not in b for b in acc.get("bureaus", []))
    assert "account_fingerprint" in acc
