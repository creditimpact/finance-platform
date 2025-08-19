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
    acc = {"action_tag": "direct_dispute", "address": "123 St"}
    populate_required_fields(acc)
    assert acc["furnisher_address"] == "123 St"

    acc2 = {"action_tag": "cease_and_desist", "name": "Collector"}
    populate_required_fields(acc2)
    assert acc2["collector_name"] == "Collector"
