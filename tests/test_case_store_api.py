import json
from pathlib import Path
from typing import Dict

import pytest

from backend.core.case_store import api, storage
from backend.core.case_store.errors import CaseStoreError, NOT_FOUND, VALIDATION_FAILED
from backend.core.case_store.models import Bureau


# Helpers ---------------------------------------------------------------

def configure(monkeypatch, tmp_path: Path, *, redact: bool = True) -> None:
    monkeypatch.setattr(storage, "CASESTORE_DIR", tmp_path.as_posix())
    monkeypatch.setattr(api, "CASESTORE_REDACT_BEFORE_STORE", redact)


def bootstrap_session(monkeypatch, tmp_path: Path, *, redact: bool = True) -> str:
    configure(monkeypatch, tmp_path, redact=redact)
    case = api.create_session_case("sess", meta={"raw_source": {"vendor": "SmartCredit"}})
    api.save_session_case(case)
    return case.session_id


# Tests ----------------------------------------------------------------

def test_round_trip(tmp_path, monkeypatch):
    configure(monkeypatch, tmp_path)
    case = api.create_session_case("abc", meta={"raw_source": {"vendor": "SmartCredit"}})
    api.save_session_case(case)
    loaded = api.load_session_case("abc")
    assert loaded.report_meta.raw_source["vendor"] == "SmartCredit"


def test_upsert_account_fields_redaction(tmp_path, monkeypatch):
    session_id = bootstrap_session(monkeypatch, tmp_path, redact=True)
    api.upsert_account_fields(
        session_id,
        "acc1",
        "Equifax",
        {
            "account_number": "1234 5678 9012",
            "balance_owed": 5000,
            "payment_status": "120D late",
        },
    )
    loaded = api.load_session_case(session_id)
    acc = loaded.accounts["acc1"]
    assert acc.bureau == Bureau.Equifax
    assert acc.fields.account_number == "****9012"
    assert acc.fields.balance_owed == 5000


def test_upsert_account_fields_no_redaction(tmp_path, monkeypatch):
    session_id = bootstrap_session(monkeypatch, tmp_path, redact=False)
    api.upsert_account_fields(
        session_id,
        "acc1",
        "Experian",
        {"account_number": "1234 5678 9012"},
    )
    loaded = api.load_session_case(session_id)
    acc = loaded.accounts["acc1"]
    assert acc.fields.account_number == "1234 5678 9012"


def test_get_account_fields_subset(tmp_path, monkeypatch):
    session_id = bootstrap_session(monkeypatch, tmp_path, redact=False)
    api.upsert_account_fields(
        session_id,
        "acc1",
        "Equifax",
        {"balance_owed": 5000, "payment_status": "late"},
    )
    fields = api.get_account_fields(
        session_id,
        "acc1",
        ["balance_owed", "payment_status", "missing_field"],
    )
    assert fields == {
        "balance_owed": 5000,
        "payment_status": "late",
        "missing_field": None,
    }


def test_append_artifact_and_overwrite(tmp_path, monkeypatch):
    session_id = bootstrap_session(monkeypatch, tmp_path, redact=False)
    api.upsert_account_fields(session_id, "acc1", "Equifax", {})
    payload: Dict[str, object] = {
        "primary_issue": "unknown",
        "confidence": 0.0,
        "tier": "none",
        "decision_source": "rules",
        "problem_reasons": [],
    }
    api.append_artifact(session_id, "acc1", "stageA_detection", payload)
    loaded = api.load_session_case(session_id)
    art = loaded.accounts["acc1"].artifacts["stageA_detection"]
    assert art.timestamp is not None
    with pytest.raises(CaseStoreError) as exc:
        api.append_artifact(session_id, "acc1", "stageA_detection", payload, overwrite=False)
    assert exc.value.code == VALIDATION_FAILED


def test_set_tags_and_list_accounts(tmp_path, monkeypatch):
    session_id = bootstrap_session(monkeypatch, tmp_path, redact=False)
    api.upsert_account_fields(session_id, "acc_eq", "Equifax", {})
    api.upsert_account_fields(session_id, "acc_ex", "Experian", {})
    api.set_tags(session_id, "acc_eq", is_problematic=True, tier="Tier2")
    loaded = api.load_session_case(session_id)
    assert loaded.accounts["acc_eq"].tags["is_problematic"] is True
    assert loaded.accounts["acc_eq"].tags["tier"] == "Tier2"
    ids_all = set(api.list_accounts(session_id))
    assert ids_all == {"acc_eq", "acc_ex"}
    ids_eq = api.list_accounts(session_id, bureau="Equifax")
    assert ids_eq == ["acc_eq"]
    with pytest.raises(CaseStoreError) as exc:
        api.list_accounts(session_id, bureau="BadBureau")
    assert exc.value.code == VALIDATION_FAILED


def test_error_paths(tmp_path, monkeypatch):
    configure(monkeypatch, tmp_path)
    with pytest.raises(CaseStoreError) as exc:
        api.get_account_case("missing", "acc")
    assert exc.value.code == NOT_FOUND

    session_id = bootstrap_session(monkeypatch, tmp_path)
    with pytest.raises(CaseStoreError) as exc:
        api.get_account_case(session_id, "missing")
    assert exc.value.code == NOT_FOUND
    with pytest.raises(CaseStoreError) as exc:
        api.append_artifact(session_id, "missing", "ns", {})
    assert exc.value.code == NOT_FOUND
    with pytest.raises(CaseStoreError) as exc:
        api.set_tags(session_id, "missing", foo=1)
    assert exc.value.code == NOT_FOUND
