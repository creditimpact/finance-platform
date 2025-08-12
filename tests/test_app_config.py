import pytest

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backend.api import config


def test_get_app_config_success(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.example.com/v1")
    cfg = config.get_app_config()
    assert cfg.ai.api_key == "key"
    assert cfg.ai.base_url == "https://api.example.com/v1"
    assert cfg.wkhtmltopdf_path == "wkhtmltopdf"
    assert cfg.smtp_server == "localhost"


def test_get_app_config_invalid(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.example.com/v1")
    with pytest.raises(EnvironmentError):
        config.get_app_config()
    monkeypatch.setenv("OPENAI_API_KEY", "key")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost")
    with pytest.raises(EnvironmentError):
        config.get_app_config()
