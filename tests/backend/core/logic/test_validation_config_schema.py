import os
from pathlib import Path

import pytest

from backend.core.logic.validation_requirements import (
    ValidationConfigError,
    load_validation_config,
)


@pytest.fixture(autouse=True)
def _clear_validation_config_cache():
    load_validation_config.cache_clear()
    yield
    load_validation_config.cache_clear()


def _write_config(path: Path, contents: str) -> Path:
    config_path = path / "config.yml"
    config_path.write_text(contents, encoding="utf-8")
    return config_path


_VALID_CONFIG = """
schema_version: 1
mode: broad
defaults:
  category: base
  min_days: 3
  points: 2
  documents: [doc1]
  strength: soft
  ai_needed: false
  min_corroboration: 1
  conditional_gate: false
fields:
  foo:
    category: base
    min_days: 3
    points: 2
    documents: [doc1]
    strength: strong
    ai_needed: false
category_defaults: {}
"""


def test_load_validation_config_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_path = _write_config(tmp_path, _VALID_CONFIG)
    monkeypatch.delenv("VALIDATION_CANARY_PERCENT", raising=False)
    config = load_validation_config(config_path)
    assert config.mode == "broad"
    assert "foo" in config.fields
    assert config.fields["foo"].strength == "strong"


def test_load_validation_config_missing_required_field(tmp_path: Path) -> None:
    invalid_config = _VALID_CONFIG.replace("ai_needed: false\n", "")
    config_path = _write_config(tmp_path, invalid_config)
    with pytest.raises(ValidationConfigError):
        load_validation_config(config_path)


def test_load_validation_config_invalid_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_path = _write_config(tmp_path, _VALID_CONFIG)
    monkeypatch.setenv("VALIDATION_CANARY_PERCENT", "250")
    with pytest.raises(ValidationConfigError):
        load_validation_config(config_path)
