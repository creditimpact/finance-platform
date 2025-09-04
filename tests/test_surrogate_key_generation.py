import pytest
from hashlib import sha1

from backend.core.case_store import api, storage
from backend.core.logic.report_analysis.extractors import accounts
from backend.core.config.flags import Flags


def setup_case(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "CASESTORE_DIR", tmp_path.as_posix())
    case = api.create_session_case("sess")
    api.save_session_case(case)
    return case.session_id


def test_surrogate_key_used_when_logical_key_absent(tmp_path, monkeypatch):
    session_id = setup_case(tmp_path, monkeypatch)
    monkeypatch.setattr(accounts, "FLAGS", Flags(one_case_per_account_enabled=True))
    monkeypatch.setattr(accounts, "compute_logical_account_key", lambda *a, **k: None)
    calls = []

    def fake_increment(name, value=1, tags=None):
        calls.append((name, tags or {}))

    monkeypatch.setattr(accounts.metrics, "increment", fake_increment)
    captured = {}

    def fake_get_or_create(sid, logical_key):
        captured["lk"] = logical_key
        return logical_key

    monkeypatch.setattr(accounts, "get_or_create_logical_account_id", fake_get_or_create)

    lines = ["Acct # ABC", "Balance Owed: $10"]
    res = accounts.extract(lines, session_id=session_id, bureau="Experian")
    assert len(res) == 1
    lk = captured["lk"]
    assert lk.startswith("surrogate_")
    assert len(lk) == len("surrogate_") + 16
    assert not any(
        name == "casebuilder.dropped" and c.get("reason") == "missing_logical_key"
        for name, c in calls
    )
    assert any(name == "casebuilder.surrogate_key_used" for name, _ in calls)


def test_surrogate_key_deterministic(tmp_path, monkeypatch):
    monkeypatch.setattr(accounts, "FLAGS", Flags(one_case_per_account_enabled=True))
    monkeypatch.setattr(accounts, "compute_logical_account_key", lambda *a, **k: None)
    monkeypatch.setattr(accounts, "get_or_create_logical_account_id", lambda s, lk: lk)

    session1 = setup_case(tmp_path, monkeypatch)
    lines = ["Acct # ABC", "Balance Owed: $10"]
    key1 = accounts.extract(lines, session_id=session1, bureau="Experian")[0]["account_id"]

    session2 = setup_case(tmp_path, monkeypatch)
    key2 = accounts.extract(lines, session_id=session2, bureau="Experian")[0]["account_id"]
    assert key1 == key2


def test_surrogate_changes_when_block_index_differs(tmp_path, monkeypatch):
    session_id = setup_case(tmp_path, monkeypatch)
    monkeypatch.setattr(accounts, "FLAGS", Flags(one_case_per_account_enabled=True))
    monkeypatch.setattr(accounts, "compute_logical_account_key", lambda *a, **k: None)
    monkeypatch.setattr(accounts, "get_or_create_logical_account_id", lambda s, lk: lk)

    lines = [
        "Acct # ABC",
        "Balance Owed: $1",
        "",
        "Acct # ABC",
        "Balance Owed: $1",
    ]
    res = accounts.extract(lines, session_id=session_id, bureau="Experian")
    assert len(res) == 2
    assert res[0]["account_id"] != res[1]["account_id"]


def test_surrogate_uses_fallback_issuer_when_heading_missing(tmp_path, monkeypatch):
    session_id = setup_case(tmp_path, monkeypatch)
    monkeypatch.setattr(accounts, "FLAGS", Flags(one_case_per_account_enabled=True))
    monkeypatch.setattr(accounts, "compute_logical_account_key", lambda *a, **k: None)
    monkeypatch.setattr(accounts, "get_or_create_logical_account_id", lambda s, lk: lk)

    lines = ["Acct # 123456", "Creditor Type: Foo Bank"]
    res = accounts.extract(lines, session_id=session_id, bureau="Experian")
    assert len(res) == 1
    key = res[0]["account_id"]
    issuer = "Foo Bank"
    first_line = accounts._digest_first_account_line(lines)
    components = f"{issuer}|0|{first_line}"
    expected = "surrogate_" + sha1(components.encode("utf-8")).hexdigest()[:16]
    assert key == expected
