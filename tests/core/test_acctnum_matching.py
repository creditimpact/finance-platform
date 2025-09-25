"""Tests for account-number matching levels and scoring side-effects."""

from __future__ import annotations

import pytest

from backend.core.logic.report_analysis import account_merge


MATCH_CASES = [
    (
        "349992********** 349992********** -34999***********",
        "349992********** 349992********** -34999***********",
        "exact",
        account_merge.POINTS_ACCTNUM_EXACT,
    ),
    ("1234-5678-9999", "123456789999", "exact", account_merge.POINTS_ACCTNUM_EXACT),
    ("****-**789012", "789012", "last6", account_merge.POINTS_ACCTNUM_LAST6),
    ("****-****-****-0423", "X X X X 0423", "last4", account_merge.POINTS_ACCTNUM_LAST4),
    ("****-****-****-1111", "....2222", "none", 0),
    (
        "0000 1234 5678 9000",
        "123456789000",
        "exact",
        account_merge.POINTS_ACCTNUM_EXACT,
    ),
]


@pytest.fixture()
def cfg() -> account_merge.MergeCfg:
    return account_merge.get_merge_cfg({})


@pytest.mark.parametrize("left,right,expected_level,expected_points", MATCH_CASES)
def test_acctnum_match_scoring(
    cfg: account_merge.MergeCfg,
    left: str,
    right: str,
    expected_level: str,
    expected_points: int,
) -> None:
    level, debug = account_merge.acctnum_match_level(left, right)
    assert level == expected_level
    assert debug["left"]["canon_mask"]
    assert debug["right"]["canon_mask"]

    bureaus_a = {"transunion": {"account_number": left}}
    bureaus_b = {"experian": {"account_number": right}}

    result = account_merge.score_pair_0_100(bureaus_a, bureaus_b, cfg)

    assert result["aux"]["account_number"]["acctnum_level"] == expected_level
    assert result["parts"]["account_number"] == expected_points
    assert result["identity_score"] == expected_points
    assert result["total"] == expected_points

    aux_payload = account_merge._build_aux_payload(result["aux"])
    assert aux_payload["acctnum_level"] == expected_level
    matched_flag = aux_payload["matched_fields"].get("account_number")
    assert matched_flag is (expected_level != "none")

    expected_triggers = set(result["triggers"])
    crosses_threshold = expected_points >= cfg.thresholds["AI_THRESHOLD"]
    assert ("total" in expected_triggers) is crosses_threshold
    if expected_level == "none":
        assert "strong:account_number" not in expected_triggers
        assert result["decision"] == "different"
    else:
        assert "strong:account_number" in expected_triggers
        assert result["decision"] == "ai"

    expected_match = expected_level != "none"
    assert result["aux"]["account_number"]["matched"] is expected_match
