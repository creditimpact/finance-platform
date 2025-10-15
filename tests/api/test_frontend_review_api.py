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
    pack_path = run_dir / "frontend" / "review" / "packs" / "acct-1.json"
    pack_payload = {"holder_name": "Jane", "questions": []}
    _write_json(pack_path, pack_payload)

    response = client.get(
        f"/api/runs/{sid}/frontend/review/accounts/acct-1",
    )
    assert response.status_code == 200
    assert response.get_json() == pack_payload


def test_frontend_review_pack_uses_manifest_path(api_client):
    client, runs_root = api_client
    sid = "S333"
    run_dir = runs_root / sid
    manifest_path = run_dir / "frontend" / "review" / "index.json"
    pack_rel_path = "frontend/accounts/acct-1/pack.json"
    manifest_payload = {
        "sid": sid,
        "stage": "review",
        "packs": [
            {
                "account_id": "acct-1",
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
    pack_payload = {"holder_name": "Client", "questions": []}
    _write_json(legacy_pack_path, pack_payload)

    response = client.get(
        f"/api/runs/{sid}/frontend/review/accounts/acct-1",
    )
    assert response.status_code == 200
    assert response.get_json() == pack_payload


def test_frontend_review_pack_missing_returns_404(api_client):
    client, runs_root = api_client
    sid = "S404"
    runs_root.joinpath(sid).mkdir(parents=True, exist_ok=True)

    response = client.get(
        f"/api/runs/{sid}/frontend/review/accounts/missing",
    )
    assert response.status_code == 404


def test_frontend_review_submit_appends_response(api_client):
    client, runs_root = api_client
    sid = "S555"
    account_id = "acct-5"
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

    stage_path = (
        runs_root
        / sid
        / "frontend"
        / "review"
        / "responses"
        / f"{account_id}.jsonl"
    )
    legacy_path = (
        runs_root / sid / "frontend" / "responses" / f"{account_id}.jsonl"
    )

    for path in (stage_path, legacy_path):
        assert path.is_file()
        lines = path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["answers"] == payload["answers"]
        assert record["client_ts"] == payload["client_ts"]
        assert record["account_id"] == account_id
