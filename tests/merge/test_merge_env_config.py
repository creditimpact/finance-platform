"""Tests for merge configuration driven by MERGE_* environment variables."""

from __future__ import annotations

import json
import os
from typing import Iterator

import pytest

from backend.config.merge_config import reset_merge_config_cache
from backend.core.logic.report_analysis.account_merge import (
    _FIELD_SEQUENCE,
    get_merge_cfg,
)


@pytest.fixture(autouse=True)
def _reset_merge_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Ensure each test observes a clean merge configuration state."""

    # Remove any MERGE_* variables carried over from the surrounding test suite so
    # the scenarios exercised here are deterministic.
    for key in list(os.environ):
        if key.startswith("MERGE_"):
            monkeypatch.delenv(key, raising=False)

    # Clear cached configuration before running the test and again afterwards so
    # subsequent tests see the correct environment driven behaviour.
    reset_merge_config_cache()
    yield
    reset_merge_config_cache()


def test_fields_override_applies_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """Custom field list from the env block should drive the merge field set."""

    monkeypatch.setenv("MERGE_ENABLED", "true")
    monkeypatch.setenv(
        "MERGE_FIELDS_OVERRIDE_JSON",
        json.dumps(["account_number", "balance_owed", "history_2y"]),
    )

    cfg = get_merge_cfg()

    assert cfg.fields == (
        "account_number",
        "balance_owed",
        "history_2y",
    )
    assert cfg.MERGE_FIELDS_OVERRIDE == (
        "account_number",
        "balance_owed",
        "history_2y",
    )


def test_custom_weights_and_thresholds(monkeypatch: pytest.MonkeyPatch) -> None:
    """Custom weights and scoring thresholds should be honoured when toggled on."""

    monkeypatch.setenv("MERGE_ENABLED", "true")
    monkeypatch.setenv("MERGE_USE_CUSTOM_WEIGHTS", "true")
    monkeypatch.setenv(
        "MERGE_WEIGHTS_JSON",
        json.dumps({"balance_owed": 0.75, "account_type": 1.5}),
    )
    monkeypatch.setenv(
        "MERGE_THRESHOLDS_JSON",
        json.dumps({"MERGE_SCORE_THRESHOLD": 88, "AUTO_MERGE_THRESHOLD": 70}),
    )

    cfg = get_merge_cfg()

    assert cfg.use_custom_weights is True
    assert cfg.MERGE_WEIGHTS["balance_owed"] == pytest.approx(0.75)
    assert cfg.MERGE_WEIGHTS["account_type"] == pytest.approx(1.5)
    assert cfg.thresholds["MERGE_SCORE_THRESHOLD"] == 88
    assert cfg.thresholds["AUTO_MERGE_THRESHOLD"] == 70
    assert cfg.MERGE_SCORE_THRESHOLD == 88


def test_disabled_merge_preserves_legacy_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """When disabled the system should fall back to the legacy field sequence."""

    monkeypatch.setenv("MERGE_ENABLED", "false")
    monkeypatch.setenv(
        "MERGE_FIELDS_OVERRIDE_JSON",
        json.dumps(["account_number", "balance_owed"]),
    )

    cfg = get_merge_cfg()

    assert cfg.fields == _FIELD_SEQUENCE
    assert cfg.MERGE_ALLOWLIST_ENFORCE is False


def test_tolerance_overrides_respected(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tolerance overrides from the env config should update the runtime values."""

    monkeypatch.setenv("MERGE_ENABLED", "true")
    monkeypatch.setenv(
        "MERGE_TOLERANCES_JSON",
        json.dumps(
            {
                "MERGE_TOL_DATE_DAYS": 5,
                "MERGE_TOL_BALANCE_ABS": 12.5,
                "MERGE_TOL_BALANCE_RATIO": 0.15,
            }
        ),
    )

    cfg = get_merge_cfg()

    assert cfg.tolerances["MERGE_TOL_DATE_DAYS"] == 5
    assert cfg.tolerances["MERGE_TOL_BALANCE_ABS"] == pytest.approx(12.5)
    assert cfg.tolerances["MERGE_TOL_BALANCE_RATIO"] == pytest.approx(0.15)


def test_default_flags_match_legacy_behaviour() -> None:
    """Without the new env flags the legacy merge configuration should remain."""

    cfg = get_merge_cfg()

    assert cfg.fields == _FIELD_SEQUENCE
    assert cfg.use_custom_weights is False
    assert cfg.MERGE_ALLOWLIST_ENFORCE is False


def test_points_mode_enforces_allowlist(monkeypatch: pytest.MonkeyPatch) -> None:
    """Points mode should honour the ENV-defined allowlist and weights only."""

    override_fields = [
        "balance_owed",
        "account_number",
        "history_7y",
        "history_2y",
        "account_status",
        "account_type",
        "date_opened",
    ]
    weights = {
        "account_number": 1.0,
        "date_opened": 1.0,
        "balance_owed": 3.0,
        "account_type": 0.5,
        "account_status": 0.5,
        "history_2y": 1.0,
        "history_7y": 1.0,
    }

    monkeypatch.setenv("MERGE_ENABLED", "true")
    monkeypatch.setenv("MERGE_POINTS_MODE", "true")
    monkeypatch.setenv("MERGE_FIELDS_OVERRIDE_JSON", json.dumps(override_fields))
    monkeypatch.setenv("MERGE_WEIGHTS_JSON", json.dumps(weights))

    cfg = get_merge_cfg()

    assert cfg.points_mode is True
    assert cfg.fields == tuple(override_fields)
    assert tuple(cfg.MERGE_FIELDS_OVERRIDE) == tuple(override_fields)
    assert all(field in cfg.MERGE_WEIGHTS for field in override_fields)
    assert cfg.MERGE_WEIGHTS == pytest.approx(weights)
