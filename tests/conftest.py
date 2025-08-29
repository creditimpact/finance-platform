import sys


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

