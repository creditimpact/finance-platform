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
    assert "frontend" in queue_names

    routes = tasks_module._flatten_task_routes(tasks_module.app.conf.task_routes)
    route_config = routes.get("backend.api.tasks.generate_frontend_packs_task")
    assert route_config is not None
    assert route_config.get("queue") == "frontend"
    assert route_config.get("routing_key") == "frontend"
