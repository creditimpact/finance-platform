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
from backend.core.ai.eligibility_policy import (
    ALWAYS_ELIGIBLE_FIELDS as POLICY_ALWAYS_ELIGIBLE_FIELDS,
    CONDITIONAL_FIELDS as POLICY_CONDITIONAL_FIELDS,
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
            "findings": [
                dict(requirement, send_to_ai=True)
                for requirement in requirements
            ],
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


def _pattern_values(field: str, pattern: str) -> dict[str, Any]:
    base_patterns: dict[str, dict[str, Any]] = {
        "case_1": {
            "transunion": "only-tu",
            "experian": "--",
            "equifax": None,
        },
        "case_2": {
            "transunion": "aligned",
            "experian": "aligned",
            "equifax": "",
        },
        "case_3": {
            "transunion": "alpha",
            "experian": "beta",
            "equifax": "--",
        },
        "case_4": {
            "transunion": "shared",
            "experian": "shared",
            "equifax": "unique",
        },
        "case_5": {
            "transunion": "one",
            "experian": "two",
            "equifax": "three",
        },
        "case_6": {
            "transunion": "",
            "experian": None,
            "equifax": "--",
        },
    }

    if field == "two_year_payment_history" and pattern == "case_5":
        return {
            "transunion": ["30", "60"],
            "experian": ["60", "30"],
            "equifax": ["ok"],
        }

    if field == "seven_year_history" and pattern == "case_6":
        return {
            "transunion": [],
            "experian": [],
            "equifax": [],
        }

    return dict(base_patterns[pattern])


@pytest.fixture
def reason_pack_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    monkeypatch.setenv("VALIDATION_REASON_ENABLED", "1")

    sid = "SID777"
    account_id = 42
    runs_root = tmp_path / "runs"

    field_patterns: dict[str, str] = {
        # Always-eligible
        "date_opened": "case_1",
        "closed_date": "case_2",
        "account_type": "case_3",
        "creditor_type": "case_4",
        "high_balance": "case_5",
        "credit_limit": "case_6",
        "term_length": "case_1",
        "payment_amount": "case_2",
        "payment_frequency": "case_3",
        "balance_owed": "case_4",
        "last_payment": "case_5",
        "past_due_amount": "case_6",
        "date_of_last_activity": "case_1",
        "account_status": "case_2",
        "payment_status": "case_3",
        "date_reported": "case_4",
        "two_year_payment_history": "case_5",
        "seven_year_history": "case_6",
        # Conditional fields
        "creditor_remarks": "case_1",
        "account_rating": "case_4",
        "account_number_display": "case_3",
    }

    requirements: list[dict[str, Any]] = []
    field_consistency: dict[str, Any] = {}
    bureaus_payload: dict[str, dict[str, Any]] = {
        "transunion": {},
        "experian": {},
        "equifax": {},
    }

    for field, pattern in field_patterns.items():
        values = _pattern_values(field, pattern)
        requirements.append(
            {
                "field": field,
                "category": FIELD_CATEGORY_MAP[field],
                "documents": _expected_documents(field),
                "strength": "weak",
                "ai_needed": True,
            }
        )
        field_consistency[field] = {"raw": values}
        for bureau, value in values.items():
            bureaus_payload[bureau][field] = value

    summary_payload = {
        "validation_requirements": {
            "findings": [
                dict(requirement, send_to_ai=True)
                for requirement in requirements
            ],
            "field_consistency": field_consistency,
        }
    }

    account_dir = runs_root / sid / "cases" / "accounts" / str(account_id)
    _write_json(account_dir / "summary.json", summary_payload)
    _write_json(account_dir / "bureaus.json", bureaus_payload)

    writer = ValidationPackWriter(sid, runs_root=runs_root)
    lines = writer.write_pack_for_account(account_id)

    pack_path = validation_packs_dir(sid, runs_root=runs_root) / validation_pack_filename_for_account(
        account_id
    )
    jsonl_entries = _read_jsonl(pack_path)

    payload_by_field = {line.payload["field"]: line.payload for line in lines}
    jsonl_by_field = {entry["field"]: entry for entry in jsonl_entries}

    return {
        "patterns": field_patterns,
        "payloads": payload_by_field,
        "jsonl": jsonl_by_field,
    }


class TestReasonMetadata:
    def test_patterns_cover_all_cases(self, reason_pack_fixture: dict[str, Any]) -> None:
        payloads = reason_pack_fixture["payloads"]
        expected_patterns = reason_pack_fixture["patterns"]

        observed: set[str] = set()
        for field, expected_pattern in expected_patterns.items():
            reason = payloads[field]["reason"]
            assert reason["pattern"] == expected_pattern
            observed.add(reason["pattern"])

        assert observed == {f"case_{idx}" for idx in range(1, 7)}

    def test_reason_flags_respect_policy(self, reason_pack_fixture: dict[str, Any]) -> None:
        payloads = reason_pack_fixture["payloads"]

        for field, payload in payloads.items():
            reason = payload["reason"]
            missing = reason["pattern"] in {"case_1", "case_2", "case_3", "case_6"}
            mismatch = reason["pattern"] in {"case_3", "case_4", "case_5"}

            assert reason["missing"] is missing
            assert reason["mismatch"] is mismatch
            assert reason["both"] is (missing and mismatch)

            if field in POLICY_ALWAYS_ELIGIBLE_FIELDS:
                assert reason["eligible"] is (missing or mismatch)
            elif field in POLICY_CONDITIONAL_FIELDS:
                assert reason["eligible"] is mismatch

        always_fields = POLICY_ALWAYS_ELIGIBLE_FIELDS
        conditional_fields = POLICY_CONDITIONAL_FIELDS

        assert always_fields.isdisjoint(conditional_fields)
        assert always_fields | conditional_fields == set(payloads)

    def test_ai_needed_only_for_conditional_mismatches(
        self, reason_pack_fixture: dict[str, Any]
    ) -> None:
        payloads = reason_pack_fixture["payloads"]

        for field, payload in payloads.items():
            reason = payload["reason"]
            ai_needed = payload["ai_needed"]
            if field in POLICY_CONDITIONAL_FIELDS:
                assert ai_needed is (reason["mismatch"] and reason["eligible"])
            else:
                assert ai_needed is False

    def test_jsonl_lines_include_reason(self, reason_pack_fixture: dict[str, Any]) -> None:
        payloads = reason_pack_fixture["payloads"]
        jsonl_by_field = reason_pack_fixture["jsonl"]

        for field, payload in payloads.items():
            jsonl_reason = jsonl_by_field[field]["reason"]
            assert jsonl_reason == payload["reason"]
            assert set(jsonl_reason["coverage"]) == {"missing_bureaus", "present_bureaus"}

