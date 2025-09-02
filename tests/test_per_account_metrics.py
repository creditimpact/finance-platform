import logging
from typing import List

import pytest

from backend.core.case_store import api, storage
from backend.core.logic.report_analysis.extractors import accounts
from backend.core.config.flags import Flags


def _setup_case(tmp_path, monkeypatch) -> str:
    monkeypatch.setattr(storage, "CASESTORE_DIR", tmp_path.as_posix())
    case = api.create_session_case("sess")
    api.save_session_case(case)
    return case.session_id


def _lines() -> List[str]:
    return [
        "Account # 123456789",
        "Creditor Type: Bank",
        "Date Opened: 2020-01-01",
        "Balance Owed: $100",
    ]


def test_emits_per_account_mode_flag_once(tmp_path, monkeypatch):
    session_id = _setup_case(tmp_path, monkeypatch)
    monkeypatch.setattr(accounts, "FLAGS", Flags(one_case_per_account_enabled=True))
    monkeypatch.setattr(accounts, "_mode_emitted", set())
    monkeypatch.setattr(accounts, "_logical_ids", {})
    metrics: list[tuple[str, float, dict]] = []
    monkeypatch.setattr(
        accounts,
        "emit_metric",
        lambda name, value, **tags: metrics.append((name, value, tags)),
    )

    accounts.extract(_lines(), session_id=session_id, bureau="Experian")
    accounts.extract(_lines(), session_id=session_id, bureau="TransUnion")

    flag_metrics = [m for m in metrics if m[0] == "stage1.per_account_mode.enabled"]
    assert flag_metrics == [("stage1.per_account_mode.enabled", 1.0, {"session_id": session_id})]


def test_emits_by_bureau_presence_on_upsert(tmp_path, monkeypatch):
    session_id = _setup_case(tmp_path, monkeypatch)
    monkeypatch.setattr(accounts, "FLAGS", Flags(one_case_per_account_enabled=True))
    monkeypatch.setattr(accounts, "_mode_emitted", set())
    monkeypatch.setattr(accounts, "_logical_ids", {})
    metrics: list[tuple[str, float, dict]] = []
    monkeypatch.setattr(
        accounts,
        "emit_metric",
        lambda name, value, **tags: metrics.append((name, value, tags)),
    )

    res = accounts.extract(_lines(), session_id=session_id, bureau="Experian")
    account_id = res[0]["account_id"]
    accounts.extract(_lines(), session_id=session_id, bureau="TransUnion")

    presence = [m for m in metrics if m[0] == "stage1.by_bureau.present"]
    assert {m[2]["bureau"] for m in presence} == {"EX", "TU"}
    for name, value, tags in presence:
        assert value == 1.0
        assert tags["session_id"] == session_id
        assert tags["account_id"] == account_id


def test_logs_collision_without_raising(tmp_path, monkeypatch, caplog):
    session_id = _setup_case(tmp_path, monkeypatch)
    monkeypatch.setattr(accounts, "FLAGS", Flags(one_case_per_account_enabled=True))
    monkeypatch.setattr(accounts, "_mode_emitted", set())
    monkeypatch.setattr(accounts, "_logical_ids", {})
    metrics: list[tuple[str, float, dict]] = []
    monkeypatch.setattr(
        accounts,
        "emit_metric",
        lambda name, value, **tags: metrics.append((name, value, tags)),
    )

    ids = iter(["id1", "id2"])

    def fake_get_or_create(session_id, logical_key):
        return next(ids)

    monkeypatch.setattr(accounts, "get_or_create_logical_account_id", fake_get_or_create)

    accounts.extract(_lines(), session_id=session_id, bureau="Experian")
    with caplog.at_level(logging.WARNING):
        accounts.extract(_lines(), session_id=session_id, bureau="Experian")
    coll = [m for m in metrics if m[0] == "stage1.logical_index.collisions"]
    assert len(coll) == 1
    assert coll[0][1] == 1.0
    assert any("logical_index_collision" in r.message for r in caplog.records)
