import importlib

import pytest

from backend.core.case_store.merge import safe_deep_merge


def _reload_api(monkeypatch, enabled: bool):
    monkeypatch.setenv("SAFE_MERGE_ENABLED", "1" if enabled else "0")
    from backend.core.config import flags

    importlib.reload(flags)
    from backend.core.case_store import api as api_module

    importlib.reload(api_module)
    return api_module


def test_merge_dict_nested():
    base = {"a": {"b": 1, "c": {"d": 2}}}
    patch = {"a": {"c": {"e": 3}}}
    result = safe_deep_merge(base, patch)
    assert result == {"a": {"b": 1, "c": {"d": 2, "e": 3}}}


def test_merge_list_by_key():
    base = {"payment_history": [
        {"date": "2024-01-01", "status": "late"},
        {"date": "2024-02-01", "status": "ok"},
    ]}
    patch = {"payment_history": [
        {"date": "2024-01-01", "status": "ok"},
        {"date": "2024-03-01", "status": "late"},
    ]}
    result = safe_deep_merge(base, patch)
    assert result["payment_history"] == [
        {"date": "2024-01-01", "status": "ok"},
        {"date": "2024-02-01", "status": "ok"},
        {"date": "2024-03-01", "status": "late"},
    ]


def test_merge_list_union_no_key():
    base = {"tags": ["A", "B"]}
    patch = {"tags": ["B", "C"]}
    result = safe_deep_merge(base, patch)
    assert result["tags"] == ["A", "B", "C"]


def test_merge_scalar_guard():
    base = {"status": "OPEN"}
    patch_empty = {"status": ""}
    patch_none = {"status": None}
    assert safe_deep_merge(base, patch_empty)["status"] == "OPEN"
    assert safe_deep_merge(base, patch_none)["status"] == "OPEN"


def test_type_mismatch_prefers_structure():
    base = {"field": "text"}
    patch = {"field": {"inner": 1}}
    result = safe_deep_merge(base, patch)
    assert result == {"field": {"inner": 1}}

    base2 = {"field": {"inner": 1}}
    patch2 = {"field": None}
    result2 = safe_deep_merge(base2, patch2)
    assert result2 == base2


def test_flag_off_keeps_old_behavior(monkeypatch):
    api = _reload_api(monkeypatch, False)
    store = {}

    def fake_load(session_id):
        return store[session_id]

    def fake_save(case):
        store[case.session_id] = case

    monkeypatch.setattr(api, "_load", fake_load)
    monkeypatch.setattr(api, "save_session_case", fake_save)

    case = api.create_session_case("sess")
    store["sess"] = case

    api.upsert_account_fields(
        "sess",
        "acc",
        "Equifax",
        {"two_year_payment_history": ["A", "B"]},
    )
    api.upsert_account_fields(
        "sess",
        "acc",
        "Equifax",
        {"two_year_payment_history": ["B", "C"]},
    )
    result = store["sess"].accounts["acc"].fields.model_dump()
    assert result["two_year_payment_history"] == ["B", "C"]


def test_flag_on_uses_safe_merge(monkeypatch):
    api = _reload_api(monkeypatch, True)
    store = {}

    def fake_load(session_id):
        return store[session_id]

    def fake_save(case):
        store[case.session_id] = case

    monkeypatch.setattr(api, "_load", fake_load)
    monkeypatch.setattr(api, "save_session_case", fake_save)

    case = api.create_session_case("sess")
    store["sess"] = case

    api.upsert_account_fields(
        "sess",
        "acc",
        "Equifax",
        {"two_year_payment_history": ["A", "B"]},
    )
    api.upsert_account_fields(
        "sess",
        "acc",
        "Equifax",
        {"two_year_payment_history": ["B", "C"]},
    )
    result = store["sess"].accounts["acc"].fields.model_dump()
    assert result["two_year_payment_history"] == ["A", "B", "C"]
