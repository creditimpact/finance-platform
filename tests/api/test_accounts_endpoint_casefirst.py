import pytest
from dataclasses import replace


@pytest.fixture
def client(monkeypatch):
    from backend.api.app import create_app

    # Provide dummy config to avoid environment dependencies
    from types import SimpleNamespace

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


def _enable_casefirst(monkeypatch):
    import backend.core.config.flags as flags

    monkeypatch.setattr(
        flags,
        "FLAGS",
        replace(flags.FLAGS, case_first_build_required=True),
    )


def test_accounts_empty_when_no_cases(client, monkeypatch):
    _enable_casefirst(monkeypatch)

    # Case Store has no accounts
    monkeypatch.setattr("backend.core.case_store.api.list_accounts", lambda sid: [])

    # Avoid calling heavy collectors
    monkeypatch.setattr(
        "backend.api.app.collect_stageA_problem_accounts", lambda sid: []
    )
    monkeypatch.setattr(
        "backend.api.app.collect_stageA_logical_accounts", lambda sid: []
    )

    resp = client.get("/api/accounts/s123")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["ok"] is True
    assert data["accounts"] == []


def test_accounts_from_collectors_only(client, monkeypatch):
    _enable_casefirst(monkeypatch)

    # Pretend Case Store already has accounts
    monkeypatch.setattr(
        "backend.core.case_store.api.list_accounts", lambda sid: ["A1"]
    )

    # Collectors return a simple account record
    monkeypatch.setattr(
        "backend.api.app.collect_stageA_problem_accounts",
        lambda sid: [{"account_id": "A1", "bureau": "EX", "primary_issue": "late"}],
    )
    monkeypatch.setattr(
        "backend.api.app.collect_stageA_logical_accounts",
        lambda sid: [{"account_id": "A1", "bureau": "EX", "primary_issue": "late"}],
    )

    resp = client.get("/api/accounts/s123")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["ok"] is True
    accounts = data["accounts"]
    assert isinstance(accounts, list) and len(accounts) >= 1
    a = accounts[0]
    assert "parser_aggregated" not in a
