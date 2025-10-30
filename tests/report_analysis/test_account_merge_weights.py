"""Tests guarding merge scoring weights from unintended changes."""

import pytest

from backend.config.merge_config import reset_merge_config_cache
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


def test_custom_weights_respected(monkeypatch) -> None:
    """When enabled via ENV the custom weights replace the defaults."""

    monkeypatch.setenv("MERGE_ENABLED", "1")
    monkeypatch.setenv("MERGE_USE_CUSTOM_WEIGHTS", "1")
    monkeypatch.setenv("MERGE_WEIGHTS_JSON", "{\"balance_owed\": 0.75}")
    reset_merge_config_cache()

    cfg = get_merge_cfg()

    assert cfg.use_custom_weights is True
    assert cfg.points == _EXPECTED_POINTS
    assert cfg.MERGE_WEIGHTS["balance_owed"] == pytest.approx(0.75)
    assert cfg.MERGE_WEIGHTS["account_type"] == pytest.approx(1.0)

    reset_merge_config_cache()


def test_custom_weights_disabled(monkeypatch) -> None:
    """Flagged off custom weights must fall back to the static snapshot."""

    monkeypatch.setenv("MERGE_ENABLED", "1")
    monkeypatch.setenv("MERGE_USE_CUSTOM_WEIGHTS", "0")
    monkeypatch.setenv("MERGE_WEIGHTS_JSON", "{\"balance_owed\": 99}")
    reset_merge_config_cache()

    cfg = get_merge_cfg()

    assert cfg.use_custom_weights is False
    assert cfg.points == _EXPECTED_POINTS
    assert cfg.MERGE_WEIGHTS["balance_owed"] == pytest.approx(1.0)

    reset_merge_config_cache()
