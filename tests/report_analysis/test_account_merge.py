import difflib

import pytest

from backend.core.logic.report_analysis.account_merge import (
    DEFAULT_CFG,
    cluster_problematic_accounts,
    decide_merge,
    score_accounts,
)


def test_score_accounts_returns_weighted_score_and_parts():
    account_a = {
        "account_number": "**1234",
        "date_opened": "01.01.2020",
        "date_of_last_activity": "15.03.2021",
        "closed_date": "05.06.2021",
        "past_due_amount": "$1,000.00",
        "balance_owed": "2000",
        "payment_status": "Collection/Chargeoff",
        "creditor": "ACME Bank",
        "remarks": "Original creditor account",
    }
    account_b = {
        "account_number": "1234",
        "date_opened": "02.01.2020",
        "date_of_last_activity": "20.03.2021",
        "closed_date": "05.06.2021",
        "past_due_amount": "$1,050.00",
        "balance_owed": "1900",
        "payment_status": "Charge-off",
        "creditor": "ACME Financial",
        "remarks": "Original creditor acct",
    }

    score, parts = score_accounts(account_a, account_b, DEFAULT_CFG)

    expected_dates = ((364 / 365) + (360 / 365) + 1.0) / 3
    expected_balance = ((1 - (50 / 1050)) + (1 - (100 / 2000))) / 2
    normalized_a = "acme bank original creditor account"
    normalized_b = "acme financial original creditor acct"
    expected_strings = difflib.SequenceMatcher(None, normalized_a, normalized_b).ratio()

    assert parts["acct_num"] == 1.0
    assert parts["dates"] == pytest.approx(expected_dates, rel=1e-3)
    assert parts["balance"] == pytest.approx(expected_balance, rel=1e-3)
    assert parts["status"] == 1.0
    assert parts["strings"] == pytest.approx(expected_strings, rel=1e-6)

    expected_score = (
        DEFAULT_CFG.weights["acct_num"] * parts["acct_num"]
        + DEFAULT_CFG.weights["dates"] * parts["dates"]
        + DEFAULT_CFG.weights["balance"] * parts["balance"]
        + DEFAULT_CFG.weights["status"] * parts["status"]
        + DEFAULT_CFG.weights["strings"] * parts["strings"]
    )
    assert score == pytest.approx(expected_score, rel=1e-6)


@pytest.mark.parametrize(
    "value,expected",
    [
        (0.82, "auto"),
        (0.55, "ai"),
        (0.29, "different"),
        (0.33, "different"),
    ],
)
def test_decide_merge_respects_thresholds(value, expected):
    assert decide_merge(value, DEFAULT_CFG) == expected


def test_cluster_problematic_accounts_builds_clusters():
    accounts = [
        {
            "account_number": "123456789",
            "date_opened": "01.01.2019",
            "date_of_last_activity": "05.05.2021",
            "past_due_amount": "1200",
            "balance_owed": "1500",
            "payment_status": "Collection/Chargeoff",
            "creditor": "ACME Bank",
            "remarks": "Original account",
        },
        {
            "account_number": "123456789",
            "date_opened": "01.01.2019",
            "date_of_last_activity": "06.05.2021",
            "past_due_amount": "1180",
            "balance_owed": "1520",
            "payment_status": "Charge-off",
            "creditor": "ACME Bank",
            "remarks": "Original account duplicate",
        },
        {
            "account_number": "987654321",
            "date_opened": "01.01.2019",
            "date_of_last_activity": "05.05.2021",
            "past_due_amount": "1180",
            "balance_owed": "1490",
            "payment_status": "Collection",
            "creditor": "ACME Collections",
            "remarks": "Legacy debt remark",
        },
    ]

    merged = cluster_problematic_accounts(accounts, DEFAULT_CFG)
    assert len(merged) == 3

    first_tag = merged[0]["merge_tag"]
    second_tag = merged[1]["merge_tag"]
    third_tag = merged[2]["merge_tag"]

    assert first_tag["group_id"] == second_tag["group_id"]
    assert first_tag["decision"] == second_tag["decision"] == "auto"
    assert first_tag["best_match"]["account_index"] == 1
    assert second_tag["best_match"]["account_index"] == 0
    assert first_tag["parts"]["acct_num"] == 1.0

    assert third_tag["group_id"] != first_tag["group_id"]
    assert third_tag["decision"] == "ai"
    assert third_tag["best_match"] is not None
    assert third_tag["best_match"]["decision"] == "ai"
    assert third_tag["score_to"][0]["score"] >= third_tag["score_to"][1]["score"]
    assert third_tag["parts"]["acct_num"] == 0.0
