import logging
from pathlib import Path
from typing import Dict

import pytest

from backend.core.case_store import api as case_store, storage
from backend.core.config.flags import Flags
from backend.core.normalize import apply
from backend.core.logic.report_analysis.extractors import accounts

REGISTRY: Dict[str, object] = {
    "version": 1,
    "bureaus": ["EQ", "TU", "EX"],
    "fields": {
        "current_balance": {
            "type": "number",
            "bureau_labels": {
                "EQ": ["Balance"],
                "TU": ["Balance"],
            },
            "coerce": "number",
        },
        "credit_limit": {
            "type": "number",
            "bureau_labels": {
                "EQ": ["Credit Limit"],
                "TU": ["Credit Limit"],
            },
            "coerce": "number",
        },
        "opened_date": {
            "type": "date",
            "bureau_labels": {
                "EX": ["Date Opened"],
            },
            "coerce": "date_iso",
        },
        "account_status": {
            "type": "string",
            "bureau_labels": {
                "EQ": ["Status"],
            },
            "coerce": "string",
            "normalize_map": {"open": ["OPEN", "Open"]},
        },
    },
}


def test_agreed_values_provenance():
    by_bureau = {"EQ": {"Balance": 100}, "TU": {"Balance": 100}}
    overlay = apply.build_normalized(by_bureau, REGISTRY)
    field = overlay["current_balance"]
    assert field["status"] == "agreed"
    assert field["value"] == 100.0
    assert field["sources"] == {"EQ": 100, "TU": 100}


def test_conflict_with_priority_tie_break():
    by_bureau = {"EQ": {"Balance": 100}, "TU": {"Balance": 120}}
    overlay = apply.build_normalized(by_bureau, REGISTRY)
    field = overlay["current_balance"]
    assert field["status"] == "conflict"
    assert field["value"] == 100.0
    assert field["sources"] == {"EQ": 100, "TU": 120}


def test_derived_date_coercion():
    by_bureau = {"EX": {"Date Opened": "03/2020"}}
    overlay = apply.build_normalized(by_bureau, REGISTRY)
    field = overlay["opened_date"]
    assert field["status"] == "derived"
    assert field["value"] == "2020-03-01"
    assert field["sources"] == {"EX": "03/2020"}


def test_missing_field():
    by_bureau = {"EQ": {}, "TU": {}, "EX": {}}
    overlay = apply.build_normalized(by_bureau, REGISTRY)
    field = overlay["credit_limit"]
    assert field["status"] == "missing"
    assert field["sources"] == {}
    assert "value" not in field


def test_registry_coverage_metrics():
    by_bureau = {
        "EQ": {"Balance": 1, "Foo": 2},
        "TU": {"Balance": 3, "Bar": 4},
    }
    pct, unmapped = apply.compute_mapping_coverage(by_bureau, REGISTRY)
    assert pct == 50.0
    assert unmapped == {"Foo": 1, "Bar": 1}


def _bootstrap(monkeypatch, tmp_path: Path, session_id: str) -> str:
    monkeypatch.setattr(storage, "CASESTORE_DIR", tmp_path.as_posix())
    case = case_store.create_session_case(session_id)
    case_store.save_session_case(case)
    return session_id


def test_flag_gates_overlay_write(tmp_path, monkeypatch):
    account_id = "7890"
    lines = ["Account 1234567890", "Balance: 100"]
    monkeypatch.setattr(apply, "load_registry", lambda path=None: REGISTRY)

    # flag off
    session_off = _bootstrap(monkeypatch, tmp_path, "sess_off")
    from backend.core.config import flags as flags_mod

    off_flags = Flags(safe_merge_enabled=True, normalized_overlay_enabled=False, case_first_build_enabled=False)
    monkeypatch.setattr(flags_mod, "FLAGS", off_flags)
    monkeypatch.setattr(accounts, "FLAGS", off_flags)
    accounts.extract(lines, session_id=session_off, bureau="Equifax")
    case = case_store.get_account_case(session_off, account_id)
    assert "normalized" not in case.fields.model_dump()

    # flag on
    session_on = _bootstrap(monkeypatch, tmp_path, "sess_on")
    case_store.upsert_account_fields(
        session_on,
        account_id,
        "Equifax",
        {"by_bureau": {"EQ": {"Balance": 100}}},
    )
    on_flags = Flags(safe_merge_enabled=True, normalized_overlay_enabled=True, case_first_build_enabled=False)
    monkeypatch.setattr(flags_mod, "FLAGS", on_flags)
    monkeypatch.setattr(accounts, "FLAGS", on_flags)
    accounts.extract(lines, session_id=session_on, bureau="Equifax")
    case2 = case_store.get_account_case(session_on, account_id)
    fields = case2.fields.model_dump()
    assert "normalized" in fields
    assert fields["normalized"]["current_balance"]["value"] == 100.0
