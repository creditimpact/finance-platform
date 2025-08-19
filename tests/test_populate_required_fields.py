from backend.core.logic.letters.utils import populate_required_fields

def test_populate_pay_for_delete_fields():
    acc = {"action_tag": "pay_for_delete", "name": "Collector"}
    strat = {"offer_terms": "50% settlement"}
    populate_required_fields(acc, strat)
    assert acc["collector_name"] == "Collector"
    assert acc["offer_terms"] == "50% settlement"


def test_populate_goodwill_fields():
    acc = {"action_tag": "goodwill"}
    strat = {"months_since_last_late": 12, "account_history_good": "Good"}
    populate_required_fields(acc, strat)
    assert acc["months_since_last_late"] == 12
    assert acc["account_history_good"] == "Good"


def test_populate_mov_fields():
    acc = {"action_tag": "mov"}
    strat = {"cra_last_result": "verified", "days_since_cra_result": 30}
    populate_required_fields(acc, strat)
    assert acc["cra_last_result"] == "verified"
    assert acc["days_since_cra_result"] == 30


def test_populate_direct_dispute_and_cease():
    acc = {"action_tag": "direct_dispute", "address": "123 St", "name": "Furnisher"}
    populate_required_fields(acc)
    assert acc["furnisher_address"] == "123 St"
    assert acc["creditor_name"] == "Furnisher"

    acc2 = {"action_tag": "cease_and_desist", "name": "Collector"}
    populate_required_fields(acc2)
    assert acc2["collector_name"] == "Collector"


def test_populate_fraud_dispute_fields():
    acc = {"action_tag": "fraud_dispute", "name": "Bank"}
    strat = {"ftc_report_id": "ABC123"}
    populate_required_fields(acc, strat)
    assert acc["creditor_name"] == "Bank"
    assert acc["is_identity_theft"] is True
    assert acc["ftc_report_id"] == "ABC123"


def test_populate_bureau_inquiry_medical_fields():
    strat = {
        "account_number_masked": "****1",
        "bureau": "Experian",
        "legal_safe_summary": "Summary",
    }

    acc_bureau = {"action_tag": "bureau_dispute", "name": "Creditor"}
    populate_required_fields(acc_bureau, strat)
    assert acc_bureau["creditor_name"] == "Creditor"
    assert acc_bureau["bureau"] == "Experian"

    acc_inquiry = {
        "action_tag": "inquiry_dispute",
        "name": "Inq Co",
        "date": "2024-01-01",
    }
    populate_required_fields(acc_inquiry, strat)
    assert acc_inquiry["inquiry_creditor_name"] == "Inq Co"
    assert acc_inquiry["inquiry_date"] == "2024-01-01"

    acc_medical = {
        "action_tag": "medical_dispute",
        "name": "Med Co",
        "status": "Unpaid",
    }
    strat_med = strat | {"amount": 100}
    populate_required_fields(acc_medical, strat_med)
    assert acc_medical["creditor_name"] == "Med Co"
    assert acc_medical["amount"] == 100
    assert acc_medical["medical_status"] == "Unpaid"
