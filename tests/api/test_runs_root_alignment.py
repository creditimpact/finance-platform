import sys
import types

sys.modules.setdefault("requests", types.ModuleType("requests"))

from backend.api import app as app_module
from backend.pipeline import runs as runs_module


def test_api_runs_root_alignment(monkeypatch):
    windows_root = r"C:\finance\runs"
    monkeypatch.setenv("RUNS_ROOT", windows_root)

    api_root = app_module._runs_root_path()
    pipeline_root = runs_module.get_runs_root()

    assert str(api_root) == str(pipeline_root)
