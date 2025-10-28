import logging

import pytest


@pytest.mark.usefixtures("monkeypatch")
def test_frontend_queue_registered(monkeypatch):
    monkeypatch.setenv("CELERY_FRONTEND_QUEUE", "frontend")
    import backend.api.tasks as tasks_module

    # Re-apply configuration after adjusting the environment for this test.
    tasks_module._ensure_frontend_queue_configuration()

    queue_names = {
        getattr(queue, "name", None) for queue in tasks_module.app.conf.task_queues or []
    }
    assert {"frontend", "note_style", "validation"}.issubset(queue_names)

    routes = tasks_module._flatten_task_routes(tasks_module.app.conf.task_routes)
    frontend_route = routes.get("backend.api.tasks.generate_frontend_packs_task")
    assert frontend_route is not None
    assert frontend_route.get("queue") == "frontend"
    assert frontend_route.get("routing_key") == "frontend"

    note_style_tasks = {
        "backend.ai.note_style.tasks.note_style_prepare_and_send_task",
        "backend.ai.note_style.tasks.note_style_send_account_task",
        "backend.ai.note_style.tasks.note_style_send_sid_task",
    }
    for task_name in note_style_tasks:
        route = routes.get(task_name)
        assert route is not None, task_name
        assert route.get("queue") == "note_style"
        assert route.get("routing_key") == "note_style"

    validation_route = routes.get("backend.pipeline.auto_ai_tasks.validation_send")
    assert validation_route is not None
    assert validation_route.get("queue") == "validation"
    assert validation_route.get("routing_key") == "validation"


def test_frontend_queue_route_overrides_existing_routes(monkeypatch):
    monkeypatch.setenv("CELERY_FRONTEND_QUEUE", "frontend")

    import backend.api.tasks as tasks_module

    # Pretend a previous configuration mapped the task to a different queue.
    tasks_module.app.conf.task_routes = {
        "backend.api.tasks.generate_frontend_packs_task": {
            "queue": "celery",
            "routing_key": "celery",
        }
    }

    tasks_module._ensure_frontend_queue_configuration()

    routes = tasks_module._flatten_task_routes(tasks_module.app.conf.task_routes)
    frontend_route = routes.get("backend.api.tasks.generate_frontend_packs_task")
    assert frontend_route is not None
    assert frontend_route.get("queue") == "frontend"
    assert frontend_route.get("routing_key") == "frontend"


def test_generate_frontend_task_logs_receipt(monkeypatch, caplog):
    monkeypatch.setenv("CELERY_FRONTEND_QUEUE", "frontend")

    import backend.api.tasks as tasks_module

    # Ensure configuration uses the updated frontend queue value.
    tasks_module._ensure_frontend_queue_configuration()

    recorded: dict[str, object] = {}

    def _fake_generate(*args, **kwargs):
        recorded["called"] = True
        return {"packs_count": 0, "built": False}

    monkeypatch.setattr(
        tasks_module, "generate_frontend_packs_for_run", _fake_generate
    )

    with caplog.at_level(logging.INFO):
        tasks_module.generate_frontend_packs_task.run("SID-123")

    assert recorded.get("called") is True
    assert any(
        "FRONTEND_TASK_RECEIVED sid=SID-123" in message for message in caplog.messages
    )


def test_generate_frontend_task_skips_manifest_when_locked(monkeypatch):
    import backend.api.tasks as tasks_module

    recorded: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def _fake_generate(*args, **kwargs):
        return {
            "status": "locked",
            "packs_count": 0,
            "built": False,
            "packs_dir": "/tmp/frontend/packs",
            "last_built_at": None,
        }

    def _fake_update(*args, **kwargs):
        recorded.append((args, kwargs))

    monkeypatch.setattr(tasks_module, "generate_frontend_packs_for_run", _fake_generate)
    monkeypatch.setattr(tasks_module, "update_manifest_frontend", _fake_update)

    tasks_module.generate_frontend_packs_task.run("SID-LOCKED")

    assert recorded == []


def test_generate_frontend_task_updates_manifest_on_success(monkeypatch):
    import backend.api.tasks as tasks_module

    recorded: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def _fake_generate(*args, **kwargs):
        return {
            "status": "success",
            "packs_count": 3,
            "built": True,
            "packs_dir": "/tmp/frontend/packs",
            "last_built_at": "2024-01-01T00:00:00Z",
        }

    def _fake_update(*args, **kwargs):
        recorded.append((args, kwargs))

    monkeypatch.setattr(tasks_module, "generate_frontend_packs_for_run", _fake_generate)
    monkeypatch.setattr(tasks_module, "update_manifest_frontend", _fake_update)

    tasks_module.generate_frontend_packs_task.run("SID-SUCCESS")

    assert len(recorded) == 1


def test_generate_frontend_task_uses_manifest_builder_when_flagged(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("FRONTEND_BUILDER_IMPL", "manifest")

    import backend.api.tasks as tasks_module

    def _legacy_should_not_run(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("legacy builder should not run when manifest flag is set")

    monkeypatch.setattr(
        tasks_module, "generate_frontend_packs_for_run", _legacy_should_not_run
    )

    recorded_builder: dict[str, object] = {}

    def _fake_manifest_builder(sid: str, manifest: dict[str, object]) -> dict[str, object]:
        recorded_builder["called"] = True
        recorded_builder["manifest"] = dict(manifest)
        return {
            "status": "success",
            "packs_count": 2,
            "built": True,
            "packs_dir": str(tmp_path / "packs"),
            "last_built_at": "2024-01-01T00:00:00Z",
        }

    monkeypatch.setattr(
        "backend.frontend.review_pack_builder.build_review_packs",
        _fake_manifest_builder,
    )

    manifest_updates: list[tuple[tuple[object, ...], dict[str, object]]] = []
    monkeypatch.setattr(
        tasks_module,
        "update_manifest_frontend",
        lambda *args, **kwargs: manifest_updates.append((args, kwargs)),
    )

    result = tasks_module.generate_frontend_packs_task.run(
        "SID-MANIFEST", runs_root=str(tmp_path)
    )

    assert recorded_builder.get("called") is True
    manifest_hint = recorded_builder.get("manifest")
    assert isinstance(manifest_hint, dict)
    assert str(manifest_hint.get("path")).endswith("SID-MANIFEST/manifest.json")
    assert result["packs_count"] == 2
    assert manifest_updates, "manifest update should be triggered"


def test_enqueue_helper_uses_frontend_queue(monkeypatch, caplog):
    monkeypatch.setenv("CELERY_FRONTEND_QUEUE", "frontend")

    import backend.api.tasks as tasks_module

    class _TaskStub:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def apply_async(self, *, args, kwargs, queue):  # type: ignore[no-untyped-def]
            self.calls.append({"args": tuple(args), "kwargs": dict(kwargs), "queue": queue})

    stub = _TaskStub()
    monkeypatch.setattr(tasks_module, "generate_frontend_packs_task", stub)

    with caplog.at_level(logging.INFO):
        tasks_module.enqueue_generate_frontend_packs("SID-999")

    assert stub.calls == [
        {
            "args": ("SID-999",),
            "kwargs": {"runs_root": None, "force": False},
            "queue": "frontend",
        }
    ]
    assert any(
        "enqueue generate_frontend_packs_task sid=SID-999 queue=frontend" in message
        for message in caplog.messages
    )
