import backend.core.logic.report_analysis.report_postprocessing as rp
import pytest


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
    assert acc["issue_types"] == ["collection"]
    assert acc["primary_issue"] == "collection"
    assert acc["status"] == "Collection"
