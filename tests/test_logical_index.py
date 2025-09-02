import warnings
from pathlib import Path

import pytest

from backend.core.case_store import api, storage

warnings.filterwarnings("ignore", category=DeprecationWarning)
pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


def configure(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(storage, "CASESTORE_DIR", tmp_path.as_posix())


def bootstrap_session(monkeypatch, tmp_path: Path) -> str:
    configure(monkeypatch, tmp_path)
    case = api.create_session_case("sess", meta={})
    api.save_session_case(case)
    return case.session_id


def test_same_logical_key_returns_same_id(tmp_path, monkeypatch):
    session_id = bootstrap_session(monkeypatch, tmp_path)
    first = api.get_or_create_logical_account_id(session_id, "key1")
    second = api.get_or_create_logical_account_id(session_id, "key1")
    assert first == second
    loaded = api.load_session_case(session_id)
    assert loaded.summary.logical_index["key1"] == first
    assert first in loaded.accounts


def test_new_logical_keys_generate_new_ids(tmp_path, monkeypatch):
    session_id = bootstrap_session(monkeypatch, tmp_path)
    aid1 = api.get_or_create_logical_account_id(session_id, "key1")
    aid2 = api.get_or_create_logical_account_id(session_id, "key2")
    assert aid1 != aid2
    loaded = api.load_session_case(session_id)
    assert loaded.summary.logical_index["key1"] == aid1
    assert loaded.summary.logical_index["key2"] == aid2


def test_cas_retry_on_conflict(tmp_path, monkeypatch):
    session_id = bootstrap_session(monkeypatch, tmp_path)
    real_load = api._load
    call_counter = {"n": 0}

    def fake_load(sid: str):
        call_counter["n"] += 1
        case = real_load(sid)
        if call_counter["n"] == 2:
            case.version += 1
        return case

    monkeypatch.setattr(api, "_load", fake_load)
    aid = api.get_or_create_logical_account_id(session_id, "key1")
    assert call_counter["n"] >= 4
    monkeypatch.setattr(api, "_load", real_load)
    loaded = api.load_session_case(session_id)
    assert loaded.summary.logical_index["key1"] == aid
    assert aid in loaded.accounts
