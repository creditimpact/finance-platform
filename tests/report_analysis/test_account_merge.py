import difflib
import logging

import pytest

from backend.core.logic.report_analysis.account_merge import (
    DEFAULT_CFG,
    acctnum_match_level,
    cluster_problematic_accounts,
    decide_merge,
    load_config_from_env,
    score_accounts,
    build_ai_decision_pack,
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

    score, parts, aux = score_accounts(account_a, account_b, DEFAULT_CFG)

    expected_dates = ((364 / 365) + (360 / 365) + 1.0) / 3
    expected_balance = ((1 - (50 / 1050)) + (1 - (100 / 2000))) / 2
    normalized_a = "acme bank original creditor account"
    normalized_b = "acme financial original creditor acct"
    expected_strings = difflib.SequenceMatcher(None, normalized_a, normalized_b).ratio()

    assert parts["acct"] == 1.0
    assert parts["dates"] == pytest.approx(expected_dates, rel=1e-3)
    assert parts["balowed"] == pytest.approx(expected_balance, rel=1e-3)
    assert parts["status"] == 1.0
    assert parts["strings"] == pytest.approx(expected_strings, rel=1e-6)
    assert aux["acctnum_level"] == "exact"
    assert aux["acctnum_masked_any"] is True

    expected_score = (
        DEFAULT_CFG.weights["acct"] * parts["acct"]
        + DEFAULT_CFG.weights["dates"] * parts["dates"]
        + DEFAULT_CFG.weights["balowed"] * parts["balowed"]
        + DEFAULT_CFG.weights["status"] * parts["status"]
        + DEFAULT_CFG.weights["strings"] * parts["strings"]
    )
    assert score == pytest.approx(expected_score, rel=1e-6)


def test_acctnum_match_level_variants():
    exact_a = {
        "acct_num_digits": "12345678",
        "acct_num_last4": "5678",
    }
    exact_b = {
        "acct_num_digits": "12345678",
        "acct_num_last4": "5678",
    }
    assert acctnum_match_level(exact_a, exact_b) == "exact"

    last4_a = {
        "acct_num_digits": "1234",
        "acct_num_last4": "1234",
    }
    last4_b = {
        "acct_num_digits": "001231234",
        "acct_num_last4": "1234",
    }
    assert acctnum_match_level(last4_a, last4_b) == "last4"

    none_a = {
        "acct_num_digits": "1234",
        "acct_num_last4": "1234",
    }
    none_b = {
        "acct_num_digits": None,
        "acct_num_last4": None,
    }
    assert acctnum_match_level(none_a, none_b) == "none"


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


def test_load_config_from_env_respects_overrides(monkeypatch):
    monkeypatch.setenv("MERGE_AUTO_MIN", "0.91")
    monkeypatch.setenv("MERGE_W_ACCT", "0.45")
    monkeypatch.setenv("MERGE_ACCTNUM_TRIGGER_AI", "last4")
    monkeypatch.setenv("MERGE_ACCTNUM_MIN_SCORE", "0.52")
    monkeypatch.setenv("MERGE_ACCTNUM_REQUIRE_MASKED", "1")

    cfg = load_config_from_env()

    assert cfg.thresholds["auto_merge_min"] == 0.91
    assert cfg.weights["acct"] == 0.45
    assert cfg.acctnum_trigger_ai == "last4"
    assert cfg.acctnum_min_score == 0.52
    assert cfg.acctnum_require_masked is True


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
    assert first_tag["parts"]["acct"] == 1.0
    assert first_tag["aux"]["acctnum_level"] == "exact"
    assert third_tag["aux"]["acctnum_level"] == "none"

    assert third_tag["group_id"] != first_tag["group_id"]
    assert third_tag["decision"] == "ai"
    assert third_tag["best_match"] is not None
    assert third_tag["best_match"]["decision"] == "ai"
    assert third_tag["score_to"][0]["score"] >= third_tag["score_to"][1]["score"]
    assert third_tag["parts"]["acct"] == 0.0


def test_acctnum_override_lifts_low_score_to_ai(monkeypatch, caplog):
    monkeypatch.setenv("MERGE_ACCTNUM_TRIGGER_AI", "exact")
    monkeypatch.setenv("MERGE_ACCTNUM_MIN_SCORE", "0.4")
    monkeypatch.setenv("MERGE_ACCTNUM_REQUIRE_MASKED", "0")

    account_a = {
        "account_number": "1234567890",
        "payment_status": "",
    }
    account_b = {
        "account_number": "1234567890",
        "payment_status": "",
    }

    score, _, aux = score_accounts(account_a, account_b, DEFAULT_CFG)
    assert score == pytest.approx(0.25)
    assert aux["acctnum_level"] == "exact"

    with caplog.at_level(
        logging.INFO, logger="backend.core.logic.report_analysis.account_merge"
    ):
        merged = cluster_problematic_accounts(
            [dict(account_a), dict(account_b)], DEFAULT_CFG, sid="acctnum-override"
        )

    first_tag = merged[0]["merge_tag"]
    second_tag = merged[1]["merge_tag"]

    assert first_tag["decision"] == "ai"
    assert second_tag["decision"] == "ai"
    assert first_tag["reasons"]["acctnum_only_triggers_ai"] is True
    assert first_tag["reasons"]["acctnum_match_level"] == "exact"
    assert first_tag["reasons"]["acctnum_masked_any"] is False

    best_first = first_tag["best_match"]
    assert best_first["decision"] == "ai"
    assert best_first["score"] == pytest.approx(0.4)
    assert best_first["reasons"]["acctnum_only_triggers_ai"] is True
    assert best_first["reasons"]["acctnum_match_level"] == "exact"
    assert best_first["reasons"]["acctnum_masked_any"] is False
    assert first_tag["aux"]["override_reasons"]["acctnum_only_triggers_ai"] is True

    override_messages = [
        record.getMessage()
        for record in caplog.records
        if "MERGE_OVERRIDE" in record.getMessage()
    ]
    assert any(
        "MERGE_OVERRIDE sid=<acctnum-override> i=<0> j=<1> kind=acctnum level=<exact>"
        " masked_any=<0> lifted_to=<0.4000>" in msg
        for msg in override_messages
    )


def test_acctnum_override_respects_mask_requirement(monkeypatch):
    monkeypatch.setenv("MERGE_ACCTNUM_TRIGGER_AI", "exact")
    monkeypatch.setenv("MERGE_ACCTNUM_REQUIRE_MASKED", "1")

    account_a = {"account_number": "1234567890"}
    account_b = {"account_number": "1234567890"}

    score, _, aux = score_accounts(account_a, account_b, DEFAULT_CFG)
    assert score == pytest.approx(0.25)
    assert aux["acctnum_level"] == "exact"
    assert aux["acctnum_masked_any"] is False

    merged = cluster_problematic_accounts(
        [dict(account_a), dict(account_b)], DEFAULT_CFG, sid="acctnum-nomask"
    )

    first_tag = merged[0]["merge_tag"]
    assert first_tag["decision"] == "different"
    best = first_tag["best_match"]
    assert best["decision"] == "different"
    assert best["score"] == pytest.approx(0.25)
    assert "reasons" not in best
    assert "override_reasons" not in first_tag.get("aux", {})


def test_build_ai_pack_includes_account_number_highlights(monkeypatch):
    monkeypatch.setenv("MERGE_ACCTNUM_TRIGGER_AI", "last4")
    monkeypatch.setenv("MERGE_ACCTNUM_MIN_SCORE", "0.4")
    monkeypatch.setenv("MERGE_ACCTNUM_REQUIRE_MASKED", "0")

    left_account = {
        "account_index": 2,
        "account_number": "M20191************",
        "balance_owed": "5912",
    }
    right_account = {
        "account_index": 3,
        "account_number": "0000000000000191",
        "balance_owed": 1750,
    }

    merged = cluster_problematic_accounts(
        [dict(left_account), dict(right_account)], DEFAULT_CFG, sid="ai-pack"
    )

    tag = merged[0]["merge_tag"]
    pack = build_ai_decision_pack(
        left_account,
        right_account,
        score=tag["best_match"]["score"],
        parts=tag["parts"],
        aux=tag["aux"],
        reasons=tag.get("reasons"),
    )

    assert pack["decision"] == "ai"
    assert pack["acctnum"] == {"level": "last4", "masked_any": True}
    assert pack["reasons"]["acctnum_only_triggers_ai"] is True
    assert pack["left"]["highlights"]["balance_owed"] == pytest.approx(5912.0)
    assert pack["left"]["highlights"]["acct_num_raw"] == "M20191************"
    assert pack["left"]["highlights"]["acct_num_last4"] == "0191"
    assert pack["right"]["highlights"]["balance_owed"] == pytest.approx(1750.0)
    assert pack["right"]["highlights"]["acct_num_raw"] == "0000000000000191"
    assert pack["right"]["highlights"]["acct_num_last4"] == "0191"
