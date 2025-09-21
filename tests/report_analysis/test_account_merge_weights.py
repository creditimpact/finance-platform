"""Tests guarding merge scoring weights from unintended changes."""

from backend.core.logic.report_analysis.account_merge import (
    get_merge_cfg,
    score_pair_0_100,
)

_EXPECTED_POINTS = {
    "balance_owed": 31,
    "account_number": 28,
    "last_payment": 12,
    "past_due_amount": 8,
    "high_balance": 6,
    "creditor_type": 3,
    "account_type": 3,
    "payment_amount": 2,
    "credit_limit": 1,
    "last_verified": 1,
    "date_of_last_activity": 2,
    "date_reported": 1,
    "date_opened": 1,
    "closed_date": 1,
}


def test_merge_points_snapshot_defaults() -> None:
    """Ensure the active merge weights match the canonical 14-field snapshot."""

    cfg = get_merge_cfg(env={})
    assert cfg.points == _EXPECTED_POINTS


def test_score_pair_does_not_mutate_points() -> None:
    """Scoring pairs must not rebalance or mutate the configured weights."""

    cfg = get_merge_cfg(env={})
    before = dict(cfg.points)

    score_pair_0_100({}, {}, cfg)

    assert cfg.points == before == _EXPECTED_POINTS
