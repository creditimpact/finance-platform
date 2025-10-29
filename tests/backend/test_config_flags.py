"""Tests for environment-backed NOTE_STYLE flags in backend.config."""

import importlib
import sys
from typing import Any


_DEF_CONFIG_MODULE = "backend.config"


def _load_config_with_env(monkeypatch: Any, **env: str) -> Any:
    """Reload ``backend.config`` after applying temporary environment overrides."""

    for key, value in env.items():
        if value is None:
            monkeypatch.delenv(key, raising=False)
        else:
            monkeypatch.setenv(key, value)

    sys.modules.pop("backend.config.note_style", None)
    sys.modules.pop(_DEF_CONFIG_MODULE, None)
    return importlib.import_module(_DEF_CONFIG_MODULE)


def test_note_style_idempotent_flag_respects_zero(monkeypatch):
    """NOTE_STYLE_IDEMPOTENT_BY_NOTE_HASH=0 disables idempotency in config."""

    try:
        config = _load_config_with_env(
            monkeypatch, NOTE_STYLE_IDEMPOTENT_BY_NOTE_HASH="0"
        )
        assert config.NOTE_STYLE_IDEMPOTENT_BY_NOTE_HASH is False
    finally:
        sys.modules.pop("backend.config.note_style", None)
        sys.modules.pop(_DEF_CONFIG_MODULE, None)


def test_note_style_skip_flag_respects_truthy(monkeypatch):
    """NOTE_STYLE_SKIP_IF_RESULT_EXISTS honors truthy environment overrides."""

    try:
        config = _load_config_with_env(monkeypatch, NOTE_STYLE_SKIP_IF_RESULT_EXISTS="1")
        assert config.NOTE_STYLE_SKIP_IF_RESULT_EXISTS is True
    finally:
        sys.modules.pop("backend.config.note_style", None)
        sys.modules.pop(_DEF_CONFIG_MODULE, None)


def test_note_style_allow_tool_calls_truthy(monkeypatch):
    """NOTE_STYLE_ALLOW_TOOL_CALLS honors truthy environment overrides."""

    try:
        config = _load_config_with_env(monkeypatch, NOTE_STYLE_ALLOW_TOOL_CALLS="1")
        assert config.NOTE_STYLE_ALLOW_TOOL_CALLS is True
    finally:
        sys.modules.pop("backend.config.note_style", None)
        sys.modules.pop(_DEF_CONFIG_MODULE, None)
