import importlib
import json
import sys
from types import SimpleNamespace

import pytest


@pytest.fixture
def api_client(tmp_path, monkeypatch):
    runs_root = tmp_path / "runs"
    monkeypatch.setenv("RUNS_ROOT", str(runs_root))

    for key in (
        "FRONTEND_STAGE_NAME",
        "FRONTEND_PACKS_STAGE_DIR",
        "FRONTEND_PACKS_DIR",
        "FRONTEND_PACKS_RESPONSES_DIR",
        "FRONTEND_PACKS_INDEX",
        "FRONTEND_RESPONSES_DIR",
    ):
        monkeypatch.delenv(key, raising=False)

    class _DummySession:
        def get(self, *args, **kwargs):  # pragma: no cover - network blocked
            raise RuntimeError("network disabled")

        def close(self):  # pragma: no cover - no resources
            pass

    dummy_requests = SimpleNamespace(
        Session=_DummySession,
        RequestException=Exception,
    )
    monkeypatch.setitem(sys.modules, "requests", dummy_requests)

    dummy_cfg = SimpleNamespace(
        ai=SimpleNamespace(api_key="test", base_url="https://example.com"),
        celery_broker_url="redis://localhost:6379/0",
        secret_key="test",
        auth_tokens=[],
        rate_limit_per_minute=60,
    )
    app_module = importlib.import_module("backend.api.app")
    monkeypatch.setattr("backend.api.app.get_app_config", lambda: dummy_cfg)

    app = app_module.create_app()
    app.config.update(TESTING=True)

    return app.test_client(), runs_root


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_frontend_review_index_returns_manifest(api_client):
    client, runs_root = api_client
    sid = "S123"
    run_dir = runs_root / sid
    manifest_path = run_dir / "frontend" / "review" / "index.json"
    payload = {"sid": sid, "stage": "review", "packs": []}
    _write_json(manifest_path, payload)

    response = client.get(f"/api/runs/{sid}/frontend/review/index")
    assert response.status_code == 200
    assert response.get_json() == payload


def test_frontend_review_pack_returns_stage_pack(api_client):
    client, runs_root = api_client
    sid = "S222"
    run_dir = runs_root / sid
    pack_path = run_dir / "frontend" / "review" / "packs" / "idx-001.json"
    pack_payload = {
        "account_id": "idx-001",
        "holder_name": "Jane",
        "primary_issue": "",
        "display": {"display_version": 1},
    }
    _write_json(pack_path, pack_payload)

    response = client.get(
        f"/api/runs/{sid}/frontend/review/accounts/idx-001",
    )
    assert response.status_code == 200
    assert response.get_json() == pack_payload


def test_frontend_review_pack_uses_manifest_path(api_client, monkeypatch):
    client, runs_root = api_client
    sid = "S333"
    run_dir = runs_root / sid
    monkeypatch.setenv("FRONTEND_PACKS_DIR", "frontend/review/custom")
    manifest_path = run_dir / "frontend" / "review" / "index.json"
    pack_rel_path = "frontend/review/packs/idx-001.json"
    manifest_payload = {
        "sid": sid,
        "stage": "review",
        "packs": [
            {
                "account_id": "idx-001",
                "path": pack_rel_path,
                "holder_name": "Client",
                "primary_issue": "",
                "bytes": 42,
                "has_questions": True,
            }
        ],
    }
    _write_json(manifest_path, manifest_payload)

    legacy_pack_path = run_dir / pack_rel_path
    pack_payload = {
        "account_id": "idx-001",
        "holder_name": "Client",
        "primary_issue": "",
        "display": {"display_version": 1},
    }
    _write_json(legacy_pack_path, pack_payload)

    response = client.get(
        f"/api/runs/{sid}/frontend/review/accounts/idx-001",
    )
    assert response.status_code == 200
    assert response.get_json() == pack_payload


def test_frontend_review_pack_missing_returns_404(api_client):
    client, runs_root = api_client
    sid = "S404"
    runs_root.joinpath(sid).mkdir(parents=True, exist_ok=True)

    response = client.get(
        f"/api/runs/{sid}/frontend/review/accounts/idx-999",
    )
    assert response.status_code == 404


def test_frontend_review_pack_invalid_account_id_returns_404(api_client):
    client, runs_root = api_client
    sid = "S405"
    runs_root.joinpath(sid).mkdir(parents=True, exist_ok=True)

    response = client.get(
        f"/api/runs/{sid}/frontend/review/accounts/acct-1",
    )
    assert response.status_code == 404


def test_frontend_review_submit_writes_response_file(api_client):
    client, runs_root = api_client
    sid = "S555"
    account_id = "idx-005"
    run_dir = runs_root / sid
    run_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "answers": {"q1": "yes"},
        "client_ts": "2024-01-01T00:00:00Z",
    }

    response = client.post(
        f"/api/runs/{sid}/frontend/review/accounts/{account_id}/answer",
        json=payload,
    )
    assert response.status_code == 200
    assert response.get_json() == {"ok": True}

    responses_dir = runs_root / sid / "frontend" / "review" / "responses"
    target_file = responses_dir / "idx-005.result.json"
    assert target_file.is_file()

    record = json.loads(target_file.read_text(encoding="utf-8"))
    assert record["answers"] == payload["answers"]
    assert record["client_ts"] == payload["client_ts"]
    assert record["account_id"] == account_id

    legacy_dir = runs_root / sid / "frontend" / "responses"
    assert not legacy_dir.exists()
