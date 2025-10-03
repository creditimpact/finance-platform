import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest

sys.modules.setdefault(
    "requests", types.SimpleNamespace(post=lambda *args, **kwargs: None)
)

from backend.ai.validation_builder import ValidationPackWriter
from backend.core.ai.paths import (
    validation_index_path,
    validation_pack_filename_for_account,
    validation_packs_dir,
)
from backend.core.logic.validation_field_sets import (
    ALL_VALIDATION_FIELDS,
    ALWAYS_INVESTIGATABLE_FIELDS,
    CONDITIONAL_FIELDS,
)


FIELD_CATEGORY_MAP: dict[str, str] = {
    # Open / Identification
    "date_opened": "open_ident",
    "closed_date": "open_ident",
    "account_type": "open_ident",
    "creditor_type": "open_ident",
    "account_number_display": "open_ident",
    # Terms
    "high_balance": "terms",
    "credit_limit": "terms",
    "term_length": "terms",
    "payment_amount": "terms",
    "payment_frequency": "terms",
    # Activity
    "balance_owed": "activity",
    "last_payment": "activity",
    "past_due_amount": "activity",
    "date_of_last_activity": "activity",
    # Status / Reporting
    "account_status": "status",
    "payment_status": "status",
    "date_reported": "status",
    "account_rating": "status",
    "creditor_remarks": "status",
    # Histories
    "two_year_payment_history": "history",
    "seven_year_history": "history",
}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _expected_documents(field: str) -> list[str]:
    if field in ALWAYS_INVESTIGATABLE_FIELDS:
        return [f"doc_{field}"]
    return [f"conditional_doc_{field}"]


@pytest.mark.parametrize("account_id", [1, 2])
def test_pack_writer_emits_all_21_fields(tmp_path: Path, account_id: int) -> None:
    sid = "SID021"
    runs_root = tmp_path / "runs"

    requirements: list[dict[str, Any]] = []
    for field in ALL_VALIDATION_FIELDS:
        requirement: dict[str, Any] = {
            "field": field,
            "category": FIELD_CATEGORY_MAP[field],
            "documents": _expected_documents(field),
            "strength": "weak" if field in ALWAYS_INVESTIGATABLE_FIELDS else "soft",
            "ai_needed": True,
        }
        if field in CONDITIONAL_FIELDS:
            requirement["conditional_gate"] = True
            requirement["min_corroboration"] = 2
        requirements.append(requirement)

    summary_payload = {
        "validation_requirements": {
            "requirements": requirements,
            "field_consistency": {},
        }
    }

    bureaus_payload = {
        bureau: {field: f"{bureau}_{field}" for field in ALL_VALIDATION_FIELDS}
        for bureau in ("transunion", "experian", "equifax")
    }

    account_dir = runs_root / sid / "cases" / "accounts" / f"{account_id}"
    _write_json(account_dir / "summary.json", summary_payload)
    _write_json(account_dir / "bureaus.json", bureaus_payload)

    writer = ValidationPackWriter(sid, runs_root=runs_root)
    lines = writer.write_pack_for_account(account_id)

    assert len(lines) == len(ALL_VALIDATION_FIELDS)

    payload_fields = [line.payload["field"] for line in lines]
    assert sorted(payload_fields) == sorted(ALL_VALIDATION_FIELDS)

    for line in lines:
        payload = line.payload
        field = payload["field"]
        assert payload["category"] == FIELD_CATEGORY_MAP[field]
        if field in CONDITIONAL_FIELDS:
            assert payload["conditional_gate"] is True
        else:
            assert not payload.get("conditional_gate")

    pack_path = validation_packs_dir(sid, runs_root=runs_root) / validation_pack_filename_for_account(
        account_id
    )
    on_disk = _read_jsonl(pack_path)
    assert len(on_disk) == len(ALL_VALIDATION_FIELDS)
    assert sorted(entry["field"] for entry in on_disk) == sorted(ALL_VALIDATION_FIELDS)

    index_payload = json.loads(
        validation_index_path(sid, runs_root=runs_root).read_text(encoding="utf-8")
    )
    packs = index_payload.get("packs", [])
    assert len(packs) == 1
    entry = packs[0]
    assert entry["account_id"] == account_id
    assert entry["lines"] == len(ALL_VALIDATION_FIELDS)

