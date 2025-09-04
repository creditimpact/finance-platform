import json
import logging
from types import SimpleNamespace

import pytest

from backend.core.case_store import api, storage
from backend.core.logic.report_analysis.extractors import accounts


def setup_case(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "CASESTORE_DIR", tmp_path.as_posix())
    case = api.create_session_case("sess")
    api.save_session_case(case)
    return case.session_id


def _ledger_payload(records):
    for rec in records:
        msg = rec.getMessage()
        if "CASEBUILDER: ledger" in msg:
            return json.loads(msg.split("CASEBUILDER: ledger ", 1)[1])
    raise AssertionError("ledger line not found")


def test_ledger_log_emitted_with_expected_fields(tmp_path, caplog, monkeypatch):
    session_id = setup_case(tmp_path, monkeypatch)
    flags = SimpleNamespace(
        one_case_per_account_enabled=False,
        normalized_overlay_enabled=False,
        casebuilder_debug=True,
        CASEBUILDER_MIN_FIELDS=0,
    )
    monkeypatch.setattr(accounts, "FLAGS", flags)
    lines = [
        "JPMCB CARD",
        "Account # 123456789",
        "Payment Status: Current",
        "High Balance: 100",
    ]
    with caplog.at_level(logging.DEBUG):
        accounts.extract(lines, session_id=session_id, bureau="TransUnion")
    payload = _ledger_payload(caplog.records)
    assert isinstance(payload["heading"], str)
    assert isinstance(payload["key_built"], bool)
    assert isinstance(payload["fields_present_count"], int)
    assert set(payload["columns_detected"].keys()) == {"tu", "xp", "eq"}
    assert all(isinstance(v, bool) for v in payload["columns_detected"].values())
    assert isinstance(payload["persisted"], bool)
    assert isinstance(payload["filename"], str)
    assert isinstance(payload["block_index"], int)
    assert isinstance(payload["session_id"], str)
    assert "lk" in payload
    if payload["lk"]:
        assert payload["lk"].startswith("...")
    assert isinstance(payload.get("weak_fields", False), bool)


def test_ledger_marks_missing_columns_but_persists(tmp_path, caplog, monkeypatch):
    session_id = setup_case(tmp_path, monkeypatch)
    flags = SimpleNamespace(
        one_case_per_account_enabled=False,
        normalized_overlay_enabled=False,
        casebuilder_debug=True,
        CASEBUILDER_MIN_FIELDS=0,
    )
    monkeypatch.setattr(accounts, "FLAGS", flags)
    monkeypatch.setattr(
        accounts, "_detect_columns", lambda bureau: {"tu": False, "xp": False, "eq": False}
    )
    lines = [
        "JPMCB CARD",
        "Account # 123456789",
        "Payment Status: Current",
    ]
    with caplog.at_level(logging.DEBUG):
        accounts.extract(lines, session_id=session_id, bureau="TransUnion")
    payload = _ledger_payload(caplog.records)
    assert payload["columns_detected"] == {"tu": False, "xp": False, "eq": False}
    assert payload["persisted"] is True


def test_ledger_weak_fields_tagged(tmp_path, caplog, monkeypatch):
    session_id = setup_case(tmp_path, monkeypatch)
    flags = SimpleNamespace(
        one_case_per_account_enabled=False,
        normalized_overlay_enabled=False,
        casebuilder_debug=True,
        CASEBUILDER_MIN_FIELDS=5,
    )
    monkeypatch.setattr(accounts, "FLAGS", flags)
    lines = [
        "BK OF AMER",
        "Account # 123456789",
        "Payment Status: Current",
    ]
    with caplog.at_level(logging.DEBUG):
        accounts.extract(lines, session_id=session_id, bureau="TransUnion")
    payload = _ledger_payload(caplog.records)
    assert payload["weak_fields"] is True
    assert payload["persisted"] is True
