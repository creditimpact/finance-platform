import importlib
import os


def test_validation_rollback_disables_pipeline(monkeypatch):
    import backend.config as config

    previous = os.environ.get("VALIDATION_ROLLBACK")
    monkeypatch.setenv("VALIDATION_ROLLBACK", "1")
    reloaded = importlib.reload(config)

    try:
        assert reloaded.VALIDATION_ROLLBACK is True
        assert reloaded.ENABLE_VALIDATION_REQUIREMENTS is False
        assert reloaded.ENABLE_VALIDATION_AI is False
    finally:
        if previous is None:
            monkeypatch.delenv("VALIDATION_ROLLBACK", raising=False)
        else:
            monkeypatch.setenv("VALIDATION_ROLLBACK", previous)
        importlib.reload(config)
