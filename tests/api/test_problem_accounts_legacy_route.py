import pytest
from dataclasses import replace
from types import SimpleNamespace


@pytest.fixture
def client(monkeypatch):
    from backend.api.app import create_app

    dummy_cfg = SimpleNamespace(
        ai=SimpleNamespace(api_key="test", base_url="https://example.com"),
        celery_broker_url="redis://localhost:6379/0",
        secret_key="test",
        auth_tokens=[],
        rate_limit_per_minute=60,
    )
    monkeypatch.setattr("backend.api.app.get_app_config", lambda: dummy_cfg)
    app = create_app()
    app.config.update(TESTING=True)
    return app.test_client()


def test_problem_accounts_legacy_disabled(client, monkeypatch):
    import backend.core.config.flags as flags

    monkeypatch.setattr(
        flags,
        "FLAGS",
        replace(flags.FLAGS, disable_parser_ui_summary=True),
    )
    resp = client.get("/api/problem_accounts")
    assert resp.status_code == 410
    assert resp.get_json() == {"ok": False, "error": "parser_first_disabled"}
