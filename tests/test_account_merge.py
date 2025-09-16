import copy
import logging

import pytest

from backend.core.logic.report_analysis.account_merge import (
    cluster_problematic_accounts,
    decide_merge,
    score_accounts,
)


@pytest.fixture
def identical_account():
    return {
        "account_number": "1234567890",
        "date_opened": "01-01-2020",
        "date_of_last_activity": "05-01-2021",
        "closed_date": "05-02-2021",
        "balance_owed": "1500",
        "past_due_amount": "300",
        "payment_status": "Charge Off",
        "account_status": "Charge Off",
        "creditor": "Example Bank",
        "remarks": "Original creditor account",
    }


def test_identical_accounts_auto_cluster(identical_account):
    account_a = copy.deepcopy(identical_account)
    account_b = copy.deepcopy(identical_account)

    score, parts = score_accounts(account_a, account_b)

    assert score >= 0.9
    for value in parts.values():
        assert value == pytest.approx(1.0)

    assert decide_merge(score) == "auto"

    merged = cluster_problematic_accounts([account_a, account_b], sid="identical-run")

    first_tag = merged[0]["merge_tag"]
    second_tag = merged[1]["merge_tag"]

    assert first_tag["decision"] == "auto"
    assert second_tag["decision"] == "auto"
    assert first_tag["group_id"] == second_tag["group_id"]
    assert first_tag["best_match"]["account_index"] == 1
    assert second_tag["best_match"]["account_index"] == 0
    assert first_tag["best_match"]["decision"] == "auto"
    assert second_tag["best_match"]["decision"] == "auto"


def test_unrelated_accounts_different_decision():
    account_a = {
        "account_number": "11112222",
        "date_opened": "01-01-2020",
        "balance_owed": "5000",
        "payment_status": "Charge Off",
        "creditor": "Acme Bank",
    }
    account_b = {
        "account_number": "99998888",
        "date_opened": "01-01-2010",
        "balance_owed": "0",
        "payment_status": "Current",
        "creditor": "Different Creditor",
    }

    score, _ = score_accounts(account_a, account_b)
    assert score < 0.2
    assert decide_merge(score) == "different"

    merged = cluster_problematic_accounts(
        [copy.deepcopy(account_a), copy.deepcopy(account_b)], sid="unrelated"
    )

    first_tag = merged[0]["merge_tag"]
    second_tag = merged[1]["merge_tag"]

    assert first_tag["decision"] == "different"
    assert second_tag["decision"] == "different"
    assert first_tag["group_id"] != second_tag["group_id"]


def test_partial_overlap_ai_decision():
    account_a = {
        "account_number": "1234567812345678",
        "date_opened": "01-01-2020",
        "balance_owed": "5000",
        "payment_status": "Charge Off",
        "creditor": "Acme Bank",
    }
    account_b = {
        "account_number": "000012345678",
        "date_opened": "01-01-2020",
        "balance_owed": 5000,
        "payment_status": "Current",
        "creditor": "Acme Collections",
    }

    score, parts = score_accounts(account_a, account_b)

    assert 0.35 <= score < 0.78
    assert parts["acct_num"] == pytest.approx(0.7)
    assert parts["dates"] == pytest.approx(1.0)
    assert parts["balance"] == pytest.approx(1.0)
    assert parts["status"] == pytest.approx(0.0)
    assert parts["strings"] == pytest.approx(0.48)

    assert decide_merge(score) == "ai"

    merged = cluster_problematic_accounts(
        [copy.deepcopy(account_a), copy.deepcopy(account_b)], sid="partial"
    )

    first_tag = merged[0]["merge_tag"]
    second_tag = merged[1]["merge_tag"]

    assert first_tag["decision"] == "ai"
    assert second_tag["decision"] == "ai"
    assert first_tag["best_match"]["decision"] == "ai"
    assert second_tag["best_match"]["decision"] == "ai"


def test_graph_clustering_transitive_auto():
    account_a = {
        "account_number": "1111 2222 3333 4444",
        "date_opened": "01-01-2020",
        "balance_owed": 1000,
        "payment_status": "Charge Off account",
        "creditor": "Acme Bank",
    }
    account_b = {
        "account_number": "9999000011114444",
        "date_opened": "01-01-2020",
        "date_of_last_activity": "05-01-2021",
        "balance_owed": 1000,
        "past_due_amount": 500,
        "payment_status": "Charge Off account now paid as agreed",
        "creditor": "Acme Bank Collections Dept",
    }
    account_c = {
        "account_number": "9999000011114444",
        "date_of_last_activity": "05-01-2021",
        "past_due_amount": 500,
        "payment_status": "Paid as agreed",
        "creditor": "Collections Department",
    }

    score_ab, _ = score_accounts(account_a, account_b)
    score_bc, _ = score_accounts(account_b, account_c)
    score_ac, _ = score_accounts(account_a, account_c)

    assert score_ab >= 0.78
    assert score_bc >= 0.78
    assert score_ac < 0.35

    merged = cluster_problematic_accounts(
        [copy.deepcopy(account_a), copy.deepcopy(account_b), copy.deepcopy(account_c)],
        sid="graph",
    )

    tags = [item["merge_tag"] for item in merged]
    group_ids = {tag["group_id"] for tag in tags}
    assert group_ids == {"g1"}
    assert all(tag["decision"] == "auto" for tag in tags)
    assert tags[0]["best_match"]["account_index"] == 1
    assert tags[1]["best_match"]["account_index"] == 2
    assert tags[2]["best_match"]["account_index"] == 1

    assert [entry["account_index"] for entry in tags[0]["score_to"]] == [1]
    assert {entry["account_index"] for entry in tags[1]["score_to"]} == {0, 2}
    assert [entry["account_index"] for entry in tags[2]["score_to"]] == [1]


def test_cluster_problematic_accounts_logs_merge_events(caplog):
    accounts = [
        {
            "account_number": "1111 2222 3333 4444",
            "date_opened": "01-01-2020",
            "balance_owed": 1000,
            "payment_status": "Charge Off account",
            "creditor": "Acme Bank",
        },
        {
            "account_number": "9999000011114444",
            "date_opened": "01-01-2020",
            "date_of_last_activity": "05-01-2021",
            "balance_owed": 1000,
            "past_due_amount": 500,
            "payment_status": "Charge Off account now paid as agreed",
            "creditor": "Acme Bank Collections Dept",
        },
        {
            "account_number": "9999000011114444",
            "date_of_last_activity": "05-01-2021",
            "past_due_amount": 500,
            "payment_status": "Paid as agreed",
            "creditor": "Collections Department",
        },
    ]

    with caplog.at_level(logging.INFO, logger="backend.core.logic.report_analysis.account_merge"):
        cluster_problematic_accounts(accounts, sid="log-test")

    merge_messages = [record.getMessage() for record in caplog.records]
    assert any("MERGE_DECISION" in msg for msg in merge_messages)
    summary_messages = [msg for msg in merge_messages if "MERGE_SUMMARY" in msg]
    assert summary_messages
    assert "skipped_pairs=<1>" in summary_messages[-1]


def test_scoring_handles_missing_fields():
    account_a = {
        "acct_num": "--",
        "date_opened": "--",
        "date_of_last_activity": "05-01-2020",
        "balance_owed": "--",
        "past_due_amount": "50",
        "account_status": "--",
        "creditor": "Sample Creditor",
        "remarks": None,
    }
    account_b = {
        "account_number": None,
        "date_opened": "05-01-2020",
        "date_of_last_activity": "--",
        "balance_owed": 0,
        "past_due_amount": "50.00 USD",
        "account_status": "Charge-Off",
        "creditor": "Sample Creditor LLC",
        "remarks": "--",
    }

    score_first, parts_first = score_accounts(account_a, account_b)
    score_second, parts_second = score_accounts(account_a, account_b)

    assert score_first == pytest.approx(score_second)
    assert parts_first == parts_second
    assert 0 < score_first < 0.35
    assert parts_first["acct_num"] == 0.0
    assert parts_first["dates"] == 0.0
    assert parts_first["balance"] == pytest.approx(1.0)
    assert parts_first["strings"] > 0.5

    merged = cluster_problematic_accounts(
        [copy.deepcopy(account_a), copy.deepcopy(account_b)], sid="missing"
    )

    first_tag = merged[0]["merge_tag"]
    second_tag = merged[1]["merge_tag"]

    assert first_tag["best_match"]["score"] == pytest.approx(score_first)
    assert second_tag["best_match"]["score"] == pytest.approx(score_first)
    assert decide_merge(score_first) == "different"
