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
