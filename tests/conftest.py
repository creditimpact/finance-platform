import sys

import pytest


def _install_stub_requests():
    mod = type(sys)("requests")

    class _StubSession:
        def __init__(self, *args: object, **kwargs: object) -> None:  # pragma: no cover - stub
            raise RuntimeError("requests stub is not functional")

    mod.Session = _StubSession  # type: ignore[attr-defined]
    mod.adapters = None  # type: ignore[attr-defined]
    mod.exceptions = type("_RequestsExceptions", (), {})()
    return mod


def _install_stub_rapidfuzz():
    class _Fuzz:
        @staticmethod
        def WRatio(a, b):
            try:
                from difflib import SequenceMatcher

                return int(SequenceMatcher(None, a or "", b or "").ratio() * 100)
            except Exception:
                return 0

    mod = type(sys)("rapidfuzz")
    mod.fuzz = _Fuzz()
    return mod


try:  # pragma: no cover - environment-dependent
    import rapidfuzz  # type: ignore
except Exception:  # pragma: no cover - provide a minimal stub for tests
    sys.modules["rapidfuzz"] = _install_stub_rapidfuzz()

try:  # pragma: no cover - environment-dependent
    import requests  # type: ignore
except Exception:  # pragma: no cover - minimal stub
    sys.modules["requests"] = _install_stub_requests()


@pytest.fixture(autouse=True)
def _note_style_stage_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NOTE_STYLE_DEBOUNCE_MS", "0")
    monkeypatch.setenv("NOTE_STYLE_PROMPT_PEPPER", "tests-note-style-pepper")

