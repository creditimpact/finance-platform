from __future__ import annotations

import pytest

from backend.core.logic.report_analysis.account_merge import (
    get_merge_cfg,
    score_pair_0_100,
)


@pytest.fixture()
def cfg():
    return get_merge_cfg({})


def _make_bureaus(**kwargs: dict[str, dict[str, object]]) -> dict[str, dict[str, object]]:
    base = {"transunion": {}, "experian": {}, "equifax": {}}
    for bureau, values in kwargs.items():
        base[bureau] = values
    return base


def test_strong_balance_owed_trigger(cfg) -> None:
    bureaus_a = _make_bureaus(transunion={"balance_owed": "1000"})
    bureaus_b = _make_bureaus(experian={"balance_owed": 1000})

    result = score_pair_0_100(bureaus_a, bureaus_b, cfg)

    assert result["total"] == cfg.points["balance_owed"]
    assert result["parts"]["balance_owed"] == cfg.points["balance_owed"]
    assert result["mid_sum"] == 0
    assert result["dates_all"] is False
    assert result["decision"] == "ai"
    assert result["conflicts"] == []
    assert result["triggers"] == ["strong:balance_owed", "total"]


ACCOUNT_POINTS = {
    "exact": 50,
    "last6": 35,
    "last5": 35,
    "last4": 25,
    "masked_match": 15,
}


@pytest.mark.parametrize(
    "left,right,expected_level",
    [
        ("1234", "001234", "last4"),
        ("XXXX1234", "99991234", "last4"),
        ("99-123456", "123456", "last6"),
        ("**** ****3000", "***3000", "exact"),
    ],
)
def test_strong_account_number_trigger_levels(
    cfg, left: str, right: str, expected_level: str
) -> None:
    bureaus_a = _make_bureaus(transunion={"account_number": left})
    bureaus_b = _make_bureaus(equifax={"account_number": right})

    result = score_pair_0_100(bureaus_a, bureaus_b, cfg)

    expected_points = ACCOUNT_POINTS[expected_level]
    assert result["total"] == expected_points
    assert result["parts"]["account_number"] == expected_points
    assert result["decision"] == "ai"
    assert result["conflicts"] == []
    expected_triggers = ["strong:account_number"]
    if expected_points >= cfg.thresholds["AI_THRESHOLD"]:
        expected_triggers.append("total")
    assert result["triggers"] == expected_triggers
    assert (
        result["aux"]["account_number"].get("acctnum_level")
        == expected_level
    )


def test_account_number_masked_without_digits_does_not_match(cfg) -> None:
    bureaus_a = _make_bureaus(transunion={"account_number": "****"})
    bureaus_b = _make_bureaus(equifax={"account_number": "XXXX"})

    result = score_pair_0_100(bureaus_a, bureaus_b, cfg)

    assert result["total"] == 0
    assert result["parts"]["account_number"] == 0
    assert result["decision"] == "different"
    assert result["triggers"] == []
    assert result["aux"]["account_number"]["matched"] is False


def test_mid_trigger(cfg) -> None:
    bureaus_a = _make_bureaus(
        transunion={
            "last_payment": "2023-01-05",
            "past_due_amount": "1000",
            "high_balance": "2000",
        }
    )
    bureaus_b = _make_bureaus(
        experian={
            "last_payment": "2023-01-10",
            "past_due_amount": 1000,
            "high_balance": 2000,
        }
    )

    result = score_pair_0_100(bureaus_a, bureaus_b, cfg)

    assert result["total"] == 26
    assert result["mid_sum"] == 26
    assert result["decision"] == "ai"
    assert result["triggers"] == ["mid", "total"]
    assert result["dates_all"] is False


def test_any_to_any_cross_bureau_match(cfg) -> None:
    bureaus_a = _make_bureaus(transunion={"high_balance": "900"})
    bureaus_b = _make_bureaus(equifax={"high_balance": 900})

    result = score_pair_0_100(bureaus_a, bureaus_b, cfg)

    assert result["total"] == cfg.points["high_balance"]
    assert result["parts"]["high_balance"] == cfg.points["high_balance"]
    assert result["decision"] == "different"
    assert result["triggers"] == []
    assert result["aux"]["high_balance"]["best_pair"] == (
        "transunion",
        "equifax",
    )


def test_dates_all_trigger(cfg) -> None:
    common = {
        "last_verified": "2023-02-01",
        "date_of_last_activity": "2023-01-15",
        "date_reported": "2023-02-10",
        "date_opened": "2020-01-01",
        "closed_date": "2023-03-01",
    }
    bureaus_a = _make_bureaus(transunion=common)
    bureaus_b = _make_bureaus(experian=common)

    result = score_pair_0_100(bureaus_a, bureaus_b, cfg)

    assert result["total"] == 6
    assert result["dates_all"] is True
    assert result["decision"] == "ai"
    assert result["triggers"] == ["dates"]


def test_total_trigger_without_mid(cfg) -> None:
    bureaus_a = _make_bureaus(
        transunion={
            "last_payment": "2023-01-01",
            "past_due_amount": 500,
            "account_type": "Credit Card",
            "last_verified": "2023-01-05",
            "date_of_last_activity": "2023-01-02",
        }
    )
    bureaus_b = _make_bureaus(
        equifax={
            "last_payment": "2023-01-04",
            "past_due_amount": "500",
            "account_type": "Credit Card",
            "last_verified": "2023-01-05",
            "date_of_last_activity": "2023-01-02",
        }
    )

    result = score_pair_0_100(bureaus_a, bureaus_b, cfg)

    assert result["total"] == 26
    assert result["mid_sum"] == 23
    assert result["triggers"] == ["total"]
    assert result["decision"] == "ai"
    assert result["dates_all"] is False


def test_auto_merge_success(cfg) -> None:
    bureaus_a = _make_bureaus(
        transunion={
            "balance_owed": "1500",
            "account_number": "1234567890123456",
            "last_payment": "2023-01-01",
        }
    )
    bureaus_b = _make_bureaus(
        experian={
            "balance_owed": 1500,
            "account_number": "1234567890123456",
            "last_payment": "2023-01-01",
        }
    )

    result = score_pair_0_100(bureaus_a, bureaus_b, cfg)

    assert result["total"] == 93
    assert result["decision"] == "auto"
    assert result["conflicts"] == []
    assert result["triggers"] == ["strong:balance_owed", "strong:account_number", "total"]


def test_auto_merge_blocked_by_last4_conflict(cfg) -> None:
    shared_values = {
        "balance_owed": "2000",
        "last_payment": "2023-01-10",
        "past_due_amount": "300",
        "high_balance": "2500",
        "creditor_type": "Bank",
        "account_type": "Installment",
        "payment_amount": "150",
        "credit_limit": "2500",
        "last_verified": "2023-01-15",
        "date_of_last_activity": "2023-01-10",
        "date_reported": "2023-01-20",
        "date_opened": "2019-06-01",
        "closed_date": "2023-02-01",
    }
    bureaus_a = _make_bureaus(transunion={**shared_values, "account_number": "1111222233334444"})
    bureaus_b = _make_bureaus(equifax={**shared_values, "account_number": "9999888877770000"})

    result = score_pair_0_100(bureaus_a, bureaus_b, cfg)

    assert result["total"] == 72
    assert result["decision"] == "ai"
    assert "acct_last4_mismatch" in result["conflicts"]
    assert result["triggers"] == ["strong:balance_owed", "mid", "dates", "total"]


def test_auto_merge_blocked_by_amount_conflict(cfg) -> None:
    bureaus_a = _make_bureaus(
        transunion={
            "balance_owed": "1200",
            "account_number": "5555666677778888",
            "last_payment": "2023-01-05",
            "past_due_amount": "100",
        }
    )
    bureaus_b = _make_bureaus(
        experian={
            "balance_owed": 1200,
            "account_number": "5555666677778888",
            "last_payment": "2023-01-05",
            "past_due_amount": "900",
        }
    )

    result = score_pair_0_100(bureaus_a, bureaus_b, cfg)

    assert result["total"] == 93
    assert result["decision"] == "ai"
    assert "amount_conflict:past_due_amount" in result["conflicts"]
    assert result["triggers"] == ["strong:balance_owed", "strong:account_number", "total"]


def test_missing_values_do_not_match(cfg) -> None:
    bureaus_a = _make_bureaus(transunion={"balance_owed": None})
    bureaus_b = _make_bureaus(equifax={"balance_owed": "--"})

    result = score_pair_0_100(bureaus_a, bureaus_b, cfg)

    assert result["total"] == 0
    assert result["parts"]["balance_owed"] == 0
    assert result["decision"] == "different"
    assert result["triggers"] == []
    assert result["conflicts"] == []
