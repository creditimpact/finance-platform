import io
from types import SimpleNamespace
import uuid

import pytest

import backend.api.app as app_module
from backend.pipeline.runs import RunManifest, persist_manifest, RUNS_ROOT_ENV


@pytest.fixture
def runs_root(tmp_path, monkeypatch):
    root = tmp_path / "runs"
    monkeypatch.setenv(RUNS_ROOT_ENV, str(root))
    return root


def test_set_artifact_persists_nested_paths(runs_root):
    manifest = RunManifest.for_sid("sid123")
    sample_file = runs_root.parent / "input" / "example.pdf"
    sample_file.parent.mkdir(parents=True, exist_ok=True)
    sample_file.write_bytes(b"data")

    manifest.set_artifact("traces.accounts_table", "report", sample_file)

    reloaded = RunManifest(manifest.path).load()
    stored = (
        reloaded.data["artifacts"]["traces"]["accounts_table"]["report"]
    )
    assert stored == str(sample_file.resolve())


def test_persist_manifest_writes_to_disk(runs_root):
    manifest = RunManifest.for_sid("sid456")
    artifact_path = runs_root.parent / "uploads" / "credit.pdf"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_bytes(b"pdf")

    persist_manifest(
        manifest,
        artifacts={"uploads.documents": {"credit_report": artifact_path}},
    )

    reloaded = RunManifest(manifest.path).load()
    stored = (
        reloaded.data["artifacts"]["uploads"]["documents"]["credit_report"]
    )
    assert stored == str(artifact_path.resolve())


def test_api_upload_updates_manifest(tmp_path, monkeypatch, runs_root):
    monkeypatch.chdir(tmp_path)

    dummy_cfg = SimpleNamespace(
        ai=SimpleNamespace(api_key="test", base_url="https://example.com/v1"),
        celery_broker_url="redis://localhost:6379/0",
        secret_key="secret",
        auth_tokens=[],
        rate_limit_per_minute=60,
    )
    monkeypatch.setattr(app_module, "get_app_config", lambda: dummy_cfg)

    class DummyTask:
        id = "task-123"

    monkeypatch.setattr(app_module, "run_full_pipeline", lambda sid: DummyTask())

    session_calls: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        app_module,
        "set_session",
        lambda sid, data: session_calls.append((sid, data)),
    )

    update_calls: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        app_module,
        "update_session",
        lambda sid, **data: update_calls.append((sid, data)),
    )

    class DummyUUID:
        hex = "session123"

        def __str__(self) -> str:
            return "session123"

    monkeypatch.setattr(uuid, "uuid4", lambda: DummyUUID())

    app = app_module.create_app()
    app.config.update(TESTING=True)
    client = app.test_client()

    response = client.post(
        "/api/upload",
        data={
            "email": "user@example.com",
            "file": (io.BytesIO(b"%PDF-1.4"), "report.pdf"),
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 202
    manifest_path = runs_root / "session123" / "manifest.json"
    assert manifest_path.exists()

    manifest = RunManifest(manifest_path).load()
    stored = manifest.data["artifacts"]["uploads"]["smartcredit_report"]
    expected = str(
        (runs_root / "session123" / "uploads" / "smartcredit_report.pdf").resolve()
    )
    assert stored == expected

    assert session_calls
    assert update_calls
