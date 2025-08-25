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


def test_assign_issue_types_detects_charge_off_from_grid_only():
    acc = {"grid_history_raw": {"Experian": "OK OK CO"}}
    rp._assign_issue_types(acc)
    assert acc["has_co_marker"] is True
    assert acc.get("co_bureaus") == ["Experian"]
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
        "grid_history_raw": {"Experian": "OK OK CO"},
    }
    rp._assign_issue_types(acc)
    assert acc["has_co_marker"] is True
    assert acc["issue_types"] == ["charge_off", "late_payment"]
    assert acc["primary_issue"] == "charge_off"
    assert acc["status"] == "Charge Off"
    assert acc.get("co_bureaus") == ["Experian"]


def test_assign_issue_types_no_co_in_grid():
    acc = {
        "late_payments": {"Experian": {"30": 1}},
        "grid_history_raw": {"Experian": "OK OK"},
    }
    rp._assign_issue_types(acc)
    assert acc.get("has_co_marker") is None or acc.get("has_co_marker") is False
    assert acc.get("co_bureaus") in (None, [])
    assert acc["issue_types"] == ["late_payment"]
    assert acc["primary_issue"] == "late_payment"


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


def test_assign_issue_types_from_status_texts_map():
    acc = {"status_texts": {"TransUnion": "Collection/Chargeoff"}}
    rp._assign_issue_types(acc)
    assert acc["primary_issue"] == "collection"
    assert acc["issue_types"] == ["collection", "charge_off"]


def test_assign_issue_types_status_texts_override_late():
    acc = {
        "status_texts": {"Experian": "Collection/Chargeoff"},
        "late_payments": {"Experian": {"30": 2}},
    }
    rp._assign_issue_types(acc)
    assert acc["primary_issue"] == "collection"
    assert acc["issue_types"] == ["collection", "charge_off", "late_payment"]


def test_assign_issue_types_charge_off_from_status_texts_only():
    acc = {"status_texts": {"Equifax": "Account charged off"}}
    rp._assign_issue_types(acc)
    assert acc["primary_issue"] == "charge_off"
    assert acc["issue_types"] == ["charge_off"]


def test_assign_issue_types_from_bureau_details_status_only():
    acc = {
        "bureau_details": {
            "Experian": {
                "account_status": "Collection/Charge-Off/Bad Debt",
                "past_due_amount": 50,
            }
        }
    }
    rp._assign_issue_types(acc)
    assert acc["primary_issue"] == "collection"
    assert acc["issue_types"] == ["collection", "charge_off"]
    assert acc["co_bureaus"] == ["Experian"]


def test_assign_issue_types_from_bureau_details_past_due_only():
    acc = {"bureau_details": {"Experian": {"past_due_amount": 25}}}
    rp._assign_issue_types(acc)
    assert acc["primary_issue"] == "late_payment"
    assert acc["issue_types"] == ["late_payment"]


def test_assign_issue_types_bureau_details_mixed_late():
    acc = {
        "bureau_details": {
            "Experian": {
                "account_status": "Collection",
                "past_due_amount": 100,
            }
        },
        "late_payments": {"30": 1},
    }
    rp._assign_issue_types(acc)
    assert acc["primary_issue"] == "collection"
    assert acc["issue_types"] == ["collection", "late_payment"]


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
