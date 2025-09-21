import importlib
import logging
import sys

import pytest

import backend.config  # noqa: F401  # ensure module is loaded before reloads


@pytest.fixture
def reload_backend_config():
    """Reload ``backend.config`` after mutating the environment."""

    def _reload():
        module = importlib.reload(sys.modules["backend.config"])
        return module

    yield _reload

    importlib.reload(sys.modules["backend.config"])


def _clear_ai_env(monkeypatch):
    for key in (
        "ENABLE_AI_ADJUDICATOR",
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "AI_MODEL",
        "AI_MODEL_ID",
        "AI_REQUEST_TIMEOUT",
        "AI_PACK_MAX_LINES_PER_SIDE",
    ):
        monkeypatch.delenv(key, raising=False)


def test_ai_config_defaults_disabled(monkeypatch, reload_backend_config):
    _clear_ai_env(monkeypatch)

    module = reload_backend_config()
    cfg = module.get_ai_adjudicator_config()

    assert cfg.enabled is False
    assert cfg.base_url == "https://api.openai.com/v1"
    assert cfg.api_key is None
    assert cfg.model == "gpt-4o-mini"
    assert cfg.request_timeout == 30
    assert cfg.pack_max_lines_per_side == 20
    assert module.ENABLE_AI_ADJUDICATOR is False


def test_ai_config_missing_key_warns(monkeypatch, reload_backend_config, caplog):
    _clear_ai_env(monkeypatch)
    monkeypatch.setenv("ENABLE_AI_ADJUDICATOR", "1")

    with caplog.at_level(logging.WARNING, logger="backend.config"):
        module = reload_backend_config()

    cfg = module.get_ai_adjudicator_config()

    assert cfg.enabled is False
    assert module.ENABLE_AI_ADJUDICATOR is False

    messages = [record.getMessage() for record in caplog.records]
    assert any(
        message.startswith("MERGE_V2_AI_DISABLED ") and "missing_api_key" in message
        for message in messages
    )


def test_ai_config_custom_values(monkeypatch, reload_backend_config):
    _clear_ai_env(monkeypatch)
    monkeypatch.setenv("ENABLE_AI_ADJUDICATOR", "true")
    monkeypatch.setenv("OPENAI_API_KEY", "  super-secret  ")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.test/v1/")
    monkeypatch.setenv("AI_MODEL", "gpt-merge-pro")
    monkeypatch.setenv("AI_REQUEST_TIMEOUT", "45")
    monkeypatch.setenv("AI_PACK_MAX_LINES_PER_SIDE", "12")

    module = reload_backend_config()
    cfg = module.get_ai_adjudicator_config()

    assert cfg.enabled is True
    assert cfg.base_url == "https://example.test/v1"
    assert cfg.api_key == "super-secret"
    assert cfg.model == "gpt-merge-pro"
    assert cfg.request_timeout == 45
    assert cfg.pack_max_lines_per_side == 12

    assert module.AI_MODEL_ID == "gpt-merge-pro"
    assert module.AI_MODEL == "gpt-merge-pro"
    assert module.OPENAI_BASE_URL == "https://example.test/v1"
    assert module.OPENAI_API_KEY == "super-secret"


def test_ai_config_invalid_values(monkeypatch, reload_backend_config, caplog):
    _clear_ai_env(monkeypatch)
    monkeypatch.setenv("ENABLE_AI_ADJUDICATOR", "1")
    monkeypatch.setenv("OPENAI_API_KEY", "token")
    monkeypatch.setenv("AI_REQUEST_TIMEOUT", "0")
    monkeypatch.setenv("AI_PACK_MAX_LINES_PER_SIDE", "3")
    monkeypatch.setenv("AI_MODEL", " ")

    with caplog.at_level(logging.WARNING, logger="backend.config"):
        module = reload_backend_config()

    cfg = module.get_ai_adjudicator_config()

    assert cfg.enabled is True
    assert cfg.request_timeout == 30
    assert cfg.pack_max_lines_per_side == 20
    assert cfg.model == "gpt-4o-mini"

    warnings = [
        record.getMessage()
        for record in caplog.records
        if record.getMessage().startswith("MERGE_V2_CONFIG_DEFAULT ")
    ]
    assert len(warnings) == 3
    for reason in {"min_1", "min_5", "empty"}:
        assert any(reason in warning for warning in warnings)
