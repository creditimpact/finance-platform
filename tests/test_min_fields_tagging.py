import logging
from types import SimpleNamespace

from backend.core.case_store import api, storage
from backend.core.logic.report_analysis.extractors import accounts


def setup_case(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "CASESTORE_DIR", tmp_path.as_posix())
    case = api.create_session_case("sess")
    api.save_session_case(case)
    return case.session_id


def set_flags(monkeypatch, **kwargs):
    flags = SimpleNamespace(
        one_case_per_account_enabled=False,
        normalized_overlay_enabled=False,
        casebuilder_debug=False,
    )
    for k, v in kwargs.items():
        setattr(flags, k, v)
    monkeypatch.setattr(accounts, "FLAGS", flags)
    return flags


def test_below_threshold_is_tagged_not_dropped(tmp_path, monkeypatch, caplog):
    session_id = setup_case(tmp_path, monkeypatch)
    set_flags(monkeypatch, CASEBUILDER_MIN_FIELDS=5)

    calls = []

    def fake_increment(name, value=1, tags=None):
        calls.append((name, tags or {}))

    monkeypatch.setattr(accounts.metrics, "increment", fake_increment)

    upserts = []

    def fake_upsert(*args, **kwargs):
        upserts.append(kwargs)

    monkeypatch.setattr(accounts, "upsert_account_fields", fake_upsert)

    lines = ["Acct # ABC", "Balance Owed: $10", "Credit Limit: $20"]

    with caplog.at_level(logging.DEBUG):
        res = accounts.extract(lines, session_id=session_id, bureau="Experian")

    assert len(upserts) == 1
    fields = res[0]["fields"]
    assert fields.get("_weak_fields") is True
    assert any(
        name == "casebuilder.tag.weak_fields" for name, _ in calls
    )
    assert not any(
        name == "casebuilder.dropped" and c.get("reason") == "min_fields"
        for name, c in calls
    )
    assert any(
        "CASEBUILDER: weak_fields" in record.getMessage()
        for record in caplog.records
    )


def test_meets_threshold_not_tagged(tmp_path, monkeypatch):
    session_id = setup_case(tmp_path, monkeypatch)
    set_flags(monkeypatch, CASEBUILDER_MIN_FIELDS=5)

    upserts = []
    monkeypatch.setattr(accounts, "upsert_account_fields", lambda *a, **k: upserts.append(k))

    lines = [
        "Acct # ABC",
        "Balance Owed: $10",
        "Credit Limit: $20",
        "High Balance: $30",
        "Date Opened: 2020-01-01",
        "Past Due Amount: $0",
    ]

    res = accounts.extract(lines, session_id=session_id, bureau="Experian")
    assert len(upserts) == 1
    fields = res[0]["fields"]
    assert "_weak_fields" not in fields


def test_threshold_zero_no_tagging(tmp_path, monkeypatch):
    session_id = setup_case(tmp_path, monkeypatch)
    set_flags(monkeypatch, CASEBUILDER_MIN_FIELDS=0)

    upserts = []
    monkeypatch.setattr(accounts, "upsert_account_fields", lambda *a, **k: upserts.append(k))

    lines = ["Acct # ABC", "Balance Owed: $10", "Credit Limit: $20"]
    res = accounts.extract(lines, session_id=session_id, bureau="Experian")
    assert len(upserts) == 1
    fields = res[0]["fields"]
    assert "_weak_fields" not in fields
