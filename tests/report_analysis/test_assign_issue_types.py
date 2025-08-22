import backend.core.logic.report_analysis.report_postprocessing as rp


def test_assign_issue_types_late_payment():
    acc = {"late_payments": {"Experian": {"30": 1}}}
    rp._assign_issue_types(acc)
    assert acc["issue_types"] == ["late_payment"]
    assert acc["status"] == "Delinquent"
    assert acc["advisor_comment"] == "Late payments detected"


def test_assign_issue_types_collection_from_flags():
    acc = {"flags": ["Collection"]}
    rp._assign_issue_types(acc)
    assert acc["issue_types"] == ["collection"]
    assert acc["status"] == "Collection"


def test_assign_issue_types_charge_off_from_flags():
    acc = {"flags": ["Charge-Off"]}
    rp._assign_issue_types(acc)
    assert acc["issue_types"] == ["charge_off"]
    assert acc["status"] == "Charge Off"
