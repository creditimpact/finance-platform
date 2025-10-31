"""Tests for merge configuration driven by MERGE_* environment variables."""

from __future__ import annotations

import json
import os
from typing import Iterator

import pytest

from backend.config.merge_config import reset_merge_config_cache
from backend.core.logic.report_analysis.account_merge import (
    _FIELD_SEQUENCE,
    _field_sequence_from_cfg,
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


def test_allowlist_enforcement_uses_env_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    """When the allowlist is enforced the field order should mirror the ENV override."""

    override_fields = (
        "account_number",
        "date_opened",
        "balance_owed",
        "account_type",
        "account_status",
        "history_2y",
        "history_7y",
    )

    monkeypatch.setenv("MERGE_ENABLED", "true")
    monkeypatch.setenv("MERGE_ALLOWLIST_ENFORCE", "true")
    monkeypatch.setenv("MERGE_FIELDS_OVERRIDE_JSON", json.dumps(list(override_fields)))

    cfg = get_merge_cfg()

    assert cfg.MERGE_ALLOWLIST_ENFORCE is True
    assert _field_sequence_from_cfg(cfg) == override_fields


def test_optional_fields_require_allowlist_and_toggle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Optional fields only participate when toggled and present in the allowlist."""

    base_allowlist = [
        "account_number",
        "date_opened",
        "balance_owed",
        "account_type",
        "account_status",
        "history_2y",
        "history_7y",
    ]

    monkeypatch.setenv("MERGE_ENABLED", "true")
    monkeypatch.setenv("MERGE_ALLOWLIST_ENFORCE", "true")
    monkeypatch.setenv("MERGE_USE_CREDITOR_NAME", "true")
    monkeypatch.setenv("MERGE_FIELDS_OVERRIDE_JSON", json.dumps(base_allowlist))

    cfg_without_optional = get_merge_cfg()

    sequence_without_optional = _field_sequence_from_cfg(cfg_without_optional)
    assert "creditor_name" not in sequence_without_optional

    # Updating the allowlist to include the optional field should surface it when toggled.
    base_allowlist.append("creditor_name")
    monkeypatch.setenv("MERGE_FIELDS_OVERRIDE_JSON", json.dumps(base_allowlist))
    reset_merge_config_cache()

    cfg_with_optional = get_merge_cfg()
    sequence_with_optional = _field_sequence_from_cfg(cfg_with_optional)

    assert "creditor_name" in sequence_with_optional


def test_optional_fields_disabled_when_toggle_off(monkeypatch: pytest.MonkeyPatch) -> None:
    """Allowlisted optional fields must stay inactive when their toggles are off."""

    base_allowlist = [
        "account_number",
        "date_opened",
        "balance_owed",
        "account_type",
        "account_status",
        "history_2y",
        "history_7y",
        "creditor_name",
        "original_creditor",
    ]

    monkeypatch.setenv("MERGE_ENABLED", "true")
    monkeypatch.setenv("MERGE_ALLOWLIST_ENFORCE", "true")
    monkeypatch.setenv("MERGE_FIELDS_OVERRIDE_JSON", json.dumps(base_allowlist))
    monkeypatch.setenv("MERGE_USE_CREDITOR_NAME", "0")
    monkeypatch.setenv("MERGE_USE_ORIGINAL_CREDITOR", "0")

    cfg = get_merge_cfg()
    sequence = _field_sequence_from_cfg(cfg)

    assert "creditor_name" not in sequence
    assert "original_creditor" not in sequence
    assert "creditor_name" not in cfg.MERGE_FIELDS_OVERRIDE
    assert "original_creditor" not in cfg.MERGE_FIELDS_OVERRIDE
