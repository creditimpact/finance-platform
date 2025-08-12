import io
import json

from backend.api import app as app_module
from backend.api.app import create_app


class DummyResult:
    def get(self, timeout=None):
        return {}


def test_start_process_success(monkeypatch, tmp_path):
    class DummyTask:
        def delay(self, *a, **k):
            return DummyResult()

    monkeypatch.setattr(app_module, "extract_problematic_accounts", DummyTask())
    called = {}

    def fake_run(client, proofs, flag):
        called["called"] = True

    monkeypatch.setattr(app_module, "run_credit_repair_process", fake_run)

    test_app = create_app()
    client = test_app.test_client()
    data = {
        "email": "a@example.com",
        "file": (io.BytesIO(b"%PDF-1.4"), "test.pdf"),
    }
    resp = client.post(
        "/api/start-process", data=data, content_type="multipart/form-data"
    )
    assert resp.status_code == 200
    payload = json.loads(resp.data)
    assert payload["status"] == "awaiting_user_explanations"
    assert not called.get("called")


def test_start_process_missing_file():
    test_app = create_app()
    client = test_app.test_client()
    resp = client.post(
        "/api/start-process", data={}, content_type="multipart/form-data"
    )
    assert resp.status_code == 400
    payload = json.loads(resp.data)
    assert "Missing file" in payload["message"]
