import ast
import importlib
import os
import sys
from pathlib import Path

MODULES = ["app", "tasks", "admin"]


def has_import_time_get_app_config(path: Path) -> bool:
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        for call in ast.walk(node):
            if isinstance(call, ast.Call):
                func = call.func
                if isinstance(func, ast.Name) and func.id == "get_app_config":
                    return True
                if isinstance(func, ast.Attribute) and func.attr == "get_app_config":
                    return True
    return False


def test_no_get_app_config_at_import_time():
    offenders = []
    for path in Path(".").rglob("*.py"):
        if has_import_time_get_app_config(path):
            offenders.append(str(path))
    assert not offenders, f"get_app_config() used at import time in: {offenders}"


class EnvGuard(dict):
    def __getitem__(self, key):  # pragma: no cover - used for guarding
        raise AssertionError(f"environment variable {key} accessed during import")

    def get(self, key, default=None):  # pragma: no cover - used for guarding
        raise AssertionError(f"environment variable {key} accessed during import")

    def __contains__(self, key):  # pragma: no cover - used for guarding
        raise AssertionError(f"environment variable {key} accessed during import")


def import_fresh(module_name: str):
    if module_name in sys.modules:
        del sys.modules[module_name]
    return importlib.import_module(module_name)


def test_imports_do_not_load_app_config(monkeypatch):
    monkeypatch.setattr(os, "environ", EnvGuard())
    monkeypatch.setattr(
        os,
        "getenv",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("environment variable accessed")
        ),
    )
    from backend.api import config

    monkeypatch.setattr(
        config,
        "get_app_config",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("get_app_config accessed")
        ),
    )

    for mod in MODULES:
        import_fresh(mod)
