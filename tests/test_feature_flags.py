import importlib

import pytest


import backend.core.config.flags as flags


def reload_flags():
    return importlib.reload(flags)


@pytest.fixture(autouse=True)
def clear_env(monkeypatch):
    for var in [
        "SAFE_MERGE_ENABLED",
        "NORMALIZED_OVERLAY_ENABLED",
        "CASE_FIRST_BUILD_ENABLED",
    ]:
        monkeypatch.delenv(var, raising=False)


def test_defaults_are_false():
    f = reload_flags()
    assert f.SAFE_MERGE_ENABLED is False
    assert f.NORMALIZED_OVERLAY_ENABLED is False
    assert f.CASE_FIRST_BUILD_ENABLED is False
    assert f.FLAGS == f.Flags(False, False, False)


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "Yes", "on", "On"])
def test_env_truthy_values(monkeypatch, value):
    for var in [
        "SAFE_MERGE_ENABLED",
        "NORMALIZED_OVERLAY_ENABLED",
        "CASE_FIRST_BUILD_ENABLED",
    ]:
        monkeypatch.setenv(var, value)
    f = reload_flags()
    assert f.SAFE_MERGE_ENABLED is True
    assert f.NORMALIZED_OVERLAY_ENABLED is True
    assert f.CASE_FIRST_BUILD_ENABLED is True
    assert f.FLAGS == f.Flags(True, True, True)


@pytest.mark.parametrize("value", ["0", "false", "no", "off", "", " "])
def test_env_falsy_values(monkeypatch, value):
    for var in [
        "SAFE_MERGE_ENABLED",
        "NORMALIZED_OVERLAY_ENABLED",
        "CASE_FIRST_BUILD_ENABLED",
    ]:
        monkeypatch.setenv(var, value)
    f = reload_flags()
    assert f.SAFE_MERGE_ENABLED is False
    assert f.NORMALIZED_OVERLAY_ENABLED is False
    assert f.CASE_FIRST_BUILD_ENABLED is False
    assert f.FLAGS == f.Flags(False, False, False)
