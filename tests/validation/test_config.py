from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


MODULE_NAME = "backend.validation.config_test_instance"
MODULE_PATH = Path("backend/validation/config.py")


def load_module():
    spec = importlib.util.spec_from_file_location(MODULE_NAME, MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def clear_env(monkeypatch):
    monkeypatch.delenv("DATE_TOL_DAYS", raising=False)
    monkeypatch.delenv("AMOUNT_TOL_ABS", raising=False)
    monkeypatch.delenv("AMOUNT_TOL_RATIO", raising=False)
    monkeypatch.delenv("PREVALIDATION_OUT_PATH_REL", raising=False)
    yield


def test_defaults_applied():
    config = load_module()

    assert config.get_date_tolerance_days() == 5
    assert config.get_amount_tolerance_abs() == 50.0
    assert config.get_amount_tolerance_ratio() == 0.01
    assert config.get_prevalidation_trace_relpath() == "trace/date_convention.json"


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("DATE_TOL_DAYS", "3")
    monkeypatch.setenv("AMOUNT_TOL_ABS", "75.5")
    monkeypatch.setenv("AMOUNT_TOL_RATIO", "0.05")
    monkeypatch.setenv("PREVALIDATION_OUT_PATH_REL", "alt/path.json")

    config = load_module()

    assert config.get_date_tolerance_days() == 3
    assert config.get_amount_tolerance_abs() == 75.5
    assert config.get_amount_tolerance_ratio() == 0.05
    assert config.get_prevalidation_trace_relpath() == "alt/path.json"


def test_invalid_values_fall_back(monkeypatch):
    monkeypatch.setenv("DATE_TOL_DAYS", "not-int")
    monkeypatch.setenv("AMOUNT_TOL_ABS", "not-float")
    monkeypatch.setenv("AMOUNT_TOL_RATIO", "")
    monkeypatch.setenv("PREVALIDATION_OUT_PATH_REL", "  ")

    config = load_module()

    assert config.get_date_tolerance_days() == 5
    assert config.get_amount_tolerance_abs() == 50.0
    assert config.get_amount_tolerance_ratio() == 0.01
    assert config.get_prevalidation_trace_relpath() == "trace/date_convention.json"

