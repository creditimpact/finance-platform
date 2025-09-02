import importlib

import pytest


def test_materializer_module_removed():
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("backend.core.materialize.account_materializer")
