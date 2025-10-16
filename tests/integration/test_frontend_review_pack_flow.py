import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


def _create_api_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
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
        def get(self, *args, **kwargs):  # pragma: no cover - safety
            raise RuntimeError("network disabled")

        def close(self):  # pragma: no cover - no-op
            pass

    dummy_requests = SimpleNamespace(Session=_DummySession, RequestException=Exception)
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


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def test_frontend_review_pack_flow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    client, runs_root = _create_api_client(tmp_path, monkeypatch)

    sid = "S900"
    account_id = "idx-001"
    run_dir = runs_root / sid
    pack_dir = run_dir / "frontend" / "review" / "packs"

    pack_payload = {
        "account_id": account_id,
        "holder_name": "Integration Tester",
        "primary_issue": "incorrect_information",
        "questions": [
            {"id": "ownership", "prompt": "Do you own this account?"},
            {"id": "recognize", "prompt": "Do you recognize this account?"},
        ],
        "display": {
            "holder_name": "Integration Tester",
            "primary_issue": "incorrect_information",
            "account_number": {"per_bureau": {"experian": "****1001", "equifax": "****1001", "transunion": "****1001"}},
            "account_type": {"per_bureau": {"experian": "Credit Card", "equifax": "Credit Card", "transunion": "Credit Card"}},
            "status": {"per_bureau": {"experian": "Open", "equifax": "Closed", "transunion": "Open"}},
            "balance_owed": {"per_bureau": {"experian": "$150", "equifax": "$0", "transunion": "$150"}},
            "date_opened": {"per_bureau": {"experian": "2022-01-02", "equifax": "2022-01-03", "transunion": "2022-01-01"}},
            "closed_date": {"per_bureau": {"experian": None, "equifax": "2023-05-01", "transunion": None}},
        },
    }

    _write_json(pack_dir / f"{account_id}.json", pack_payload)

    manifest_payload = {
        "sid": sid,
        "stage": "review",
        "schema_version": "1.0",
        "packs_count": 1,
        "counts": {"packs": 1, "responses": 0},
        "packs": [
            {
                "account_id": account_id,
                "holder_name": "Integration Tester",
                "primary_issue": "incorrect_information",
                "file": f"frontend/review/packs/{account_id}.json",
                "path": f"frontend/review/packs/{account_id}.json",
            }
        ],
        "responses_dir": "frontend/review/responses",
    }
    _write_json(run_dir / "frontend" / "review" / "index.json", manifest_payload)

    listing_resp = client.get(f"/api/runs/{sid}/frontend/review/packs")
    assert listing_resp.status_code == 200
    assert listing_resp.get_json() == {
        "items": [{"account_id": account_id, "file": f"frontend/review/packs/{account_id}.json"}]
    }

    pack_resp = client.get(f"/api/runs/{sid}/frontend/review/pack/{account_id}")
    assert pack_resp.status_code == 200
    assert pack_resp.get_json()["account_id"] == account_id

    submit_payload = {"answers": {"ownership": "yes"}}
    submit_resp = client.post(
        f"/api/runs/{sid}/frontend/review/response/{account_id}",
        json=submit_payload,
    )
    assert submit_resp.status_code == 200
    stored_path = run_dir / "frontend" / "review" / "responses" / f"{account_id}.result.json"
    assert stored_path.is_file()

    stored_record = json.loads(stored_path.read_text(encoding="utf-8"))
    assert stored_record["answers"] == submit_payload["answers"]
    assert stored_record["sid"] == sid
    assert stored_record["account_id"] == account_id

