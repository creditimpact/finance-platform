import logging

import pytest

from backend import config
from backend.pipeline.runs import RunManifest, persist_manifest
from backend.runflow.umbrella import schedule_note_style_after_validation


class DummyTask:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def delay(self, *args: object, **kwargs: object) -> None:
        self.calls.append((args, kwargs))


@pytest.fixture
def dummy_prepare_task(monkeypatch: pytest.MonkeyPatch) -> DummyTask:
    import backend.ai.note_style.tasks as note_style_tasks

    dummy = DummyTask()
    monkeypatch.setattr(
        note_style_tasks,
        "note_style_prepare_and_send_task",
        dummy,
        raising=False,
    )
    return dummy


@pytest.mark.usefixtures("dummy_prepare_task")
def test_schedule_skips_without_validation_completion(tmp_path, monkeypatch):
    monkeypatch.setenv("RUNS_ROOT", str(tmp_path))
    sid = "SID-validation-pending"
    run_dir = tmp_path / sid

    manifest = RunManifest.load_or_create(run_dir / "manifest.json", sid)
    persist_manifest(manifest)

    packs_dir = (run_dir / config.NOTE_STYLE_PACKS_DIR)
    packs_dir.mkdir(parents=True, exist_ok=True)
    (packs_dir / "acc_001.jsonl").write_text("{}", encoding="utf-8")

    schedule_note_style_after_validation(sid, run_dir=run_dir)

    import backend.ai.note_style.tasks as note_style_tasks

    assert note_style_tasks.note_style_prepare_and_send_task.calls == []


@pytest.mark.usefixtures("dummy_prepare_task")
def test_schedule_logs_when_no_packs(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv("RUNS_ROOT", str(tmp_path))
    sid = "SID-wait-for-packs"
    run_dir = tmp_path / sid

    manifest = RunManifest.load_or_create(run_dir / "manifest.json", sid)
    validation_status = manifest.ensure_ai_stage_status("validation")
    validation_status["sent"] = True
    validation_status["completed_at"] = "2024-01-01T00:00:00Z"
    persist_manifest(manifest)

    caplog.set_level(logging.INFO)
    schedule_note_style_after_validation(sid, run_dir=run_dir)

    import backend.ai.note_style.tasks as note_style_tasks

    assert note_style_tasks.note_style_prepare_and_send_task.calls == []
    assert any(
        "NOTE_STYLE_WAITING_FOR_PACKS" in record.getMessage() for record in caplog.records
    )


@pytest.mark.usefixtures("dummy_prepare_task")
def test_schedule_sends_when_packs_present(tmp_path, monkeypatch):
    monkeypatch.setenv("RUNS_ROOT", str(tmp_path))
    sid = "SID-send-packs"
    run_dir = tmp_path / sid

    manifest = RunManifest.load_or_create(run_dir / "manifest.json", sid)
    validation_status = manifest.ensure_ai_stage_status("validation")
    validation_status["sent"] = True
    validation_status["completed_at"] = "2024-01-01T00:00:00Z"
    persist_manifest(manifest)

    packs_dir = (run_dir / config.NOTE_STYLE_PACKS_DIR)
    packs_dir.mkdir(parents=True, exist_ok=True)
    (packs_dir / "acc_002.jsonl").write_text("{}", encoding="utf-8")

    schedule_note_style_after_validation(sid, run_dir=run_dir)

    import backend.ai.note_style.tasks as note_style_tasks

    assert note_style_tasks.note_style_prepare_and_send_task.calls == [
        ((sid,), {"runs_root": str(tmp_path)})
    ]
