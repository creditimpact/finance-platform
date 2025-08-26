from pathlib import Path

import pytest

import backend.core.case_store.storage as storage
from backend.core.case_store import (
    IO_ERROR,
    NOT_FOUND,
    VALIDATION_FAILED,
    CaseStoreError,
    ReportMeta,
    SessionCase,
    Summary,
)


def make_case(session_id: str = "sess") -> SessionCase:
    return SessionCase(
        session_id=session_id,
        accounts={},
        summary=Summary(),
        report_meta=ReportMeta(),
    )


def configure(monkeypatch, tmp_path: Path, *, atomic=True, validate=True):
    monkeypatch.setattr(storage, "CASESTORE_DIR", tmp_path.as_posix())
    monkeypatch.setattr(storage, "CASESTORE_ATOMIC_WRITES", atomic)
    monkeypatch.setattr(storage, "CASESTORE_VALIDATE_ON_LOAD", validate)


def test_save_load_round_trip(tmp_path, monkeypatch):
    configure(monkeypatch, tmp_path)
    case = make_case("abc")
    storage.save_session_case(case)
    loaded = storage.load_session_case("abc")
    assert loaded == case


def test_atomic_write_creates_no_temp(tmp_path, monkeypatch):
    configure(monkeypatch, tmp_path, atomic=True)
    case = make_case("atom")
    storage.save_session_case(case)
    assert (tmp_path / "atom.json").exists()
    assert not list(tmp_path.glob("*.tmp*"))


def test_validation_on_load(monkeypatch, tmp_path):
    configure(monkeypatch, tmp_path, validate=True)
    path = tmp_path / "bad.json"
    path.write_text('{"session": "x"}', encoding="utf-8")
    with pytest.raises(CaseStoreError) as exc:
        storage.load_session_case("bad")
    assert exc.value.code == VALIDATION_FAILED

    configure(monkeypatch, tmp_path, validate=False)
    loaded = storage.load_session_case("bad")
    assert isinstance(loaded, SessionCase)
    assert not hasattr(loaded, "session_id")


def test_missing_file(monkeypatch, tmp_path):
    configure(monkeypatch, tmp_path)
    with pytest.raises(CaseStoreError) as exc:
        storage.load_session_case("missing")
    assert exc.value.code == NOT_FOUND


def test_io_error_on_save(monkeypatch, tmp_path):
    configure(monkeypatch, tmp_path)

    def fail_open(*args, **kwargs):
        raise PermissionError("denied")

    monkeypatch.setattr("builtins.open", fail_open)
    case = make_case("io")
    with pytest.raises(CaseStoreError) as exc:
        storage.save_session_case(case)
    assert exc.value.code == IO_ERROR


def test_non_atomic_mode(tmp_path, monkeypatch):
    configure(monkeypatch, tmp_path, atomic=False)
    case = make_case("plain")
    storage.save_session_case(case)
    assert (tmp_path / "plain.json").exists()
    loaded = storage.load_session_case("plain")
    assert loaded.session_id == "plain"
