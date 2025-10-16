import importlib
import json
import os
import sys
import time
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


def _next_chunk(iterator):
    try:
        return next(iterator)
    except StopIteration:  # pragma: no cover - defensive
        raise AssertionError("stream ended unexpectedly")


def test_frontend_manifest_endpoint_supports_filter(api_client):
    client, runs_root = api_client
    sid = "S123"
    run_dir = runs_root / sid
    manifest_payload = {
        "sid": sid,
        "status": "in_progress",
        "frontend": {
            "index": "frontend/review/index.json",
            "packs_dir": "frontend/review/packs",
            "packs_count": 2,
            "results_dir": "frontend/review/responses",
        },
        "other": {"value": 1},
    }
    _write_json(run_dir / "manifest.json", manifest_payload)

    full_response = client.get(f"/api/runs/{sid}/frontend/manifest")
    assert full_response.status_code == 200
    assert full_response.get_json() == manifest_payload

    subset_response = client.get(
        f"/api/runs/{sid}/frontend/manifest", query_string={"section": "frontend"}
    )
    assert subset_response.status_code == 200
    expected_review = dict(manifest_payload["frontend"])
    expected_review["responses_dir"] = "frontend/review/responses"
    assert subset_response.get_json() == {
        "sid": sid,
        "frontend": {"review": expected_review},
    }


def test_frontend_manifest_section_preserves_existing_review_block(api_client):
    client, runs_root = api_client
    sid = "S125"
    run_dir = runs_root / sid
    manifest_payload = {
        "sid": sid,
        "frontend": {
            "review": {
                "index": "frontend/review/index.json",
                "packs_dir": "frontend/review/packs",
                "responses_dir": "frontend/review/responses",
                "packs_count": 3,
                "extra": {"note": "keep"},
            },
            "status": "success",
        },
    }
    _write_json(run_dir / "manifest.json", manifest_payload)

    subset_response = client.get(
        f"/api/runs/{sid}/frontend/manifest", query_string={"section": "frontend"}
    )
    assert subset_response.status_code == 200
    assert subset_response.get_json() == {
        "sid": sid,
        "frontend": {
            "review": {
                "index": "frontend/review/index.json",
                "packs_dir": "frontend/review/packs",
                "responses_dir": "frontend/review/responses",
                "packs_count": 3,
                "extra": {"note": "keep"},
            },
            "status": "success",
        },
    }


def test_frontend_index_returns_payload(api_client):
    client, runs_root = api_client
    sid = "S124"
    run_dir = runs_root / sid
    manifest_path = run_dir / "frontend" / "review" / "index.json"
    payload = {
        "sid": sid,
        "stage": "review",
        "items": [
            {"account_id": "idx-001", "file": "frontend/review/packs/idx-001.json"}
        ],
    }
    _write_json(manifest_path, payload)

    response = client.get(f"/api/runs/{sid}/frontend/index")
    assert response.status_code == 200
    assert response.get_json() == payload


def test_frontend_review_stream_emits_packs_ready(api_client, monkeypatch):
    client, runs_root = api_client
    sid = "S890"
    index_path = runs_root / sid / "frontend" / "review" / "index.json"
    _write_json(index_path, {"packs_count": 3})

    monkeypatch.setattr("backend.api.app._REVIEW_STREAM_QUEUE_WAIT_SECONDS", 0.05)
    monkeypatch.setattr("backend.api.app._REVIEW_STREAM_KEEPALIVE_INTERVAL", 0.1)

    response = client.get(
        f"/api/runs/{sid}/frontend/review/stream",
        buffered=False,
        headers={"Accept": "text/event-stream"},
    )
    assert response.status_code == 200
    assert response.mimetype == "text/event-stream"

    stream = response.response
    chunk = _next_chunk(stream)
    text = chunk.decode("utf-8")
    assert "event: packs_ready" in text
    assert '"packs_count": 3' in text

    response.close()


def test_frontend_review_packs_listing_from_index(api_client):
    client, runs_root = api_client
    sid = "S130"
    run_dir = runs_root / sid
    index_path = run_dir / "frontend" / "review" / "index.json"
    payload = {
        "items": [
            {"account_id": "idx-001", "file": "frontend/review/packs/idx-001.json"},
            {"account_id": "idx-002", "filename": "idx-002.json"},
        ]
    }
    _write_json(index_path, payload)

    response = client.get(f"/api/runs/{sid}/frontend/review/packs")
    assert response.status_code == 200
    data = response.get_json()
    assert data == {
        "items": [
            {"account_id": "idx-001", "file": "frontend/review/packs/idx-001.json"},
            {"account_id": "idx-002", "file": "frontend/review/packs/idx-002.json"},
        ]
    }


def test_runs_last_returns_latest_from_index(api_client):
    client, runs_root = api_client
    runs_root.mkdir(parents=True, exist_ok=True)
    _write_json(
        runs_root / "index.json",
        {
            "runs": [
                {"sid": "old", "created_at": "2025-01-01T00:00:00Z"},
                {"sid": "new", "created_at": "2025-02-01T00:00:00Z"},
            ]
        },
    )
    (runs_root / "old").mkdir(parents=True, exist_ok=True)
    (runs_root / "new").mkdir(parents=True, exist_ok=True)

    response = client.get("/api/runs/last")
    assert response.status_code == 200
    assert response.get_json() == {"sid": "new"}


def test_runs_last_falls_back_to_directory_mtime(api_client):
    client, runs_root = api_client
    first = runs_root / "A001"
    second = runs_root / "A002"
    first.mkdir(parents=True, exist_ok=True)
    second.mkdir(parents=True, exist_ok=True)

    older = time.time() - 120
    newer = older + 30
    os.utime(first, (older, older))
    os.utime(second, (newer, newer))

    response = client.get("/api/runs/last")
    assert response.status_code == 200
    assert response.get_json() == {"sid": "A002"}


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

    response = client.get(f"/api/runs/{sid}/frontend/review/pack/idx-001")
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

    response = client.get(f"/api/runs/{sid}/frontend/review/pack/idx-001")
    assert response.status_code == 200
    assert response.get_json() == pack_payload


def test_frontend_review_pack_missing_returns_404(api_client):
    client, runs_root = api_client
    sid = "S404"
    runs_root.joinpath(sid).mkdir(parents=True, exist_ok=True)

    response = client.get(f"/api/runs/{sid}/frontend/review/pack/idx-999")
    assert response.status_code == 404


def test_frontend_review_pack_invalid_account_id_returns_404(api_client):
    client, runs_root = api_client
    sid = "S405"
    runs_root.joinpath(sid).mkdir(parents=True, exist_ok=True)

    response = client.get(f"/api/runs/{sid}/frontend/review/pack/acct-1")
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
        f"/api/runs/{sid}/frontend/review/response/{account_id}",
        json=payload,
    )
    assert response.status_code == 200
    body = response.get_json()
    assert body["answers"] == payload["answers"]
    assert body["account_id"] == account_id
    assert body["sid"] == sid
    assert "received_at" in body

    responses_dir = runs_root / sid / "frontend" / "review" / "responses"
    target_file = responses_dir / "idx-005.result.json"
    assert target_file.is_file()

    record = json.loads(target_file.read_text(encoding="utf-8"))
    assert record["answers"] == payload["answers"]
    assert record["client_ts"] == payload["client_ts"]
    assert record["account_id"] == account_id

    legacy_dir = runs_root / sid / "frontend" / "responses"
    assert not legacy_dir.exists()


def test_frontend_review_submit_honors_stage_responses_override(api_client, monkeypatch):
    client, runs_root = api_client
    sid = "S778"
    account_id = "idx-006"
    run_dir = runs_root / sid
    run_dir.mkdir(parents=True, exist_ok=True)

    custom_rel = "frontend/review/custom_responses"
    monkeypatch.setenv("FRONTEND_PACKS_RESPONSES_DIR", custom_rel)

    payload = {"answers": {"q1": "custom"}}

    response = client.post(
        f"/api/runs/{sid}/frontend/review/response/{account_id}",
        json=payload,
    )
    assert response.status_code == 200

    responses_dir = run_dir / custom_rel
    stored_path = responses_dir / "idx-006.result.json"
    assert stored_path.is_file()

    stored = json.loads(stored_path.read_text(encoding="utf-8"))
    assert stored["answers"] == payload["answers"]
    assert stored["account_id"] == account_id

    default_path = run_dir / "frontend" / "review" / "responses" / "idx-006.result.json"
    assert not default_path.exists()


def test_frontend_review_submit_accepts_client_meta(api_client, monkeypatch):
    client, runs_root = api_client
    sid = "S777"
    account_id = "idx-007"
    run_dir = runs_root / sid
    run_dir.mkdir(parents=True, exist_ok=True)

    fixed_now = "2024-02-02T10:00:00"
    monkeypatch.setattr("backend.api.app._now_utc_iso", lambda: fixed_now)

    payload = {
        "answers": {"q1": "yes"},
        "client_meta": {"browser": "Chrome"},
    }

    response = client.post(
        f"/api/runs/{sid}/frontend/review/response/{account_id}",
        json=payload,
    )
    assert response.status_code == 200
    record = response.get_json()
    assert record == {
        "sid": sid,
        "account_id": account_id,
        "answers": payload["answers"],
        "client_meta": payload["client_meta"],
        "received_at": fixed_now,
    }

    stored = json.loads(
        (runs_root / sid / "frontend" / "review" / "responses" / "idx-007.result.json")
        .read_text(encoding="utf-8")
    )
    assert stored == record


def test_frontend_review_stream_emits_responses_written_event(api_client, monkeypatch):
    client, runs_root = api_client
    sid = "S891"
    account_id = "idx-010"
    index_path = runs_root / sid / "frontend" / "review" / "index.json"
    _write_json(index_path, {"packs_count": 1})

    monkeypatch.setattr("backend.api.app._REVIEW_STREAM_QUEUE_WAIT_SECONDS", 0.05)
    monkeypatch.setattr("backend.api.app._REVIEW_STREAM_KEEPALIVE_INTERVAL", 0.1)

    response = client.get(
        f"/api/runs/{sid}/frontend/review/stream",
        buffered=False,
        headers={"Accept": "text/event-stream"},
    )

    stream = response.response
    _next_chunk(stream)  # consume packs_ready event

    payload = {"answers": {"status": "ok"}}
    post_resp = client.post(
        f"/api/runs/{sid}/frontend/review/response/{account_id}",
        json=payload,
    )
    assert post_resp.status_code == 200

    chunk = _next_chunk(stream)
    text = chunk.decode("utf-8")
    assert "event: responses_written" in text
    assert account_id in text

    response.close()
