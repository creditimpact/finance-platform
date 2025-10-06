import json
from pathlib import Path

import pytest

from backend.ai.validation_builder import ValidationPackWriter
from backend.ai.validation_results import store_validation_result
from backend.core.ai.paths import (
    validation_pack_filename_for_account,
    validation_packs_dir,
    validation_result_jsonl_filename_for_account,
    validation_result_summary_filename_for_account,
    validation_results_dir,
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _build_summary(*requirements: dict[str, object]) -> dict[str, object]:
    return {
        "validation_requirements": {
            "findings": [dict(req) for req in requirements],
            "field_consistency": {},
        }
    }


def test_pack_built_from_summary_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sid = "SID100"
    account_id = 1
    runs_root = tmp_path / "runs"

    finding = {
        "field": "account_type",
        "is_mismatch": True,
        "ai_needed": True,
        "send_to_ai": True,
        "bureau_values": {
            "transunion": {"raw": "installment"},
            "experian": {"raw": "installment"},
            "equifax": {"raw": "installment"},
        },
    }

    account_dir = runs_root / sid / "cases" / "accounts" / str(account_id)
    _write_json(account_dir / "summary.json", _build_summary(finding))

    def _fail_load_bureaus(self: ValidationPackWriter, account: int) -> dict[str, object]:  # type: ignore[override]
        raise AssertionError("bureaus.json should not be read when summary already has bureau_values")

    monkeypatch.setattr(ValidationPackWriter, "_load_bureaus", _fail_load_bureaus)

    writer = ValidationPackWriter(sid, runs_root=runs_root)
    lines = writer.write_pack_for_account(account_id)

    assert len(lines) == 1
    payload = lines[0].payload
    assert payload["finding"] == finding

    pack_path = (
        validation_packs_dir(sid, runs_root=runs_root)
        / validation_pack_filename_for_account(account_id)
    )
    assert pack_path.exists()


def test_excluded_fields_no_pack(tmp_path: Path) -> None:
    sid = "SID200"
    account_id = 2
    runs_root = tmp_path / "runs"

    summary_payload = _build_summary(
        {
            "field": "seven_year_history",
            "is_mismatch": True,
            "ai_needed": True,
            "send_to_ai": True,
        },
        {
            "field": "account_number_display",
            "is_mismatch": True,
            "ai_needed": True,
            "send_to_ai": True,
        },
    )

    account_dir = runs_root / sid / "cases" / "accounts" / str(account_id)
    _write_json(account_dir / "summary.json", summary_payload)

    writer = ValidationPackWriter(sid, runs_root=runs_root)
    lines = writer.write_pack_for_account(account_id)

    assert lines == []


def test_two_year_history_fallback(tmp_path: Path) -> None:
    sid = "SID300"
    account_id = 3
    runs_root = tmp_path / "runs"

    summary_payload = _build_summary(
        {
            "field": "two_year_payment_history",
            "is_mismatch": True,
            "ai_needed": False,
            "send_to_ai": False,
        }
    )

    account_dir = runs_root / sid / "cases" / "accounts" / str(account_id)
    _write_json(account_dir / "summary.json", summary_payload)

    writer = ValidationPackWriter(sid, runs_root=runs_root)
    lines = writer.write_pack_for_account(account_id)

    assert len(lines) == 1
    assert lines[0].payload["field"] == "two_year_payment_history"


def test_single_result_file(tmp_path: Path) -> None:
    sid = "SID400"
    account_id = 4
    runs_root = tmp_path / "runs"

    summary_payload = _build_summary(
        {
            "field": "account_type",
            "is_mismatch": True,
            "ai_needed": True,
            "send_to_ai": True,
        },
        {
            "field": "account_rating",
            "is_mismatch": True,
            "ai_needed": True,
            "send_to_ai": True,
        },
    )

    account_dir = runs_root / sid / "cases" / "accounts" / str(account_id)
    _write_json(account_dir / "summary.json", summary_payload)

    writer = ValidationPackWriter(sid, runs_root=runs_root)
    lines = writer.write_pack_for_account(account_id)
    assert len(lines) == 2
    expected_ids = [str(line.payload.get("id")) for line in lines]

    response_payload = {
        "decision_per_field": [
            {
                "field": "account_type",
                "decision": "strong",
                "rationale": "values diverge",
            },
            {
                "field": "account_rating",
                "decision": "no_case",
                "rationale": "matches reporting",
            },
        ]
    }

    result_path = store_validation_result(
        sid,
        account_id,
        response_payload,
        runs_root=runs_root,
        status="done",
    )

    results_dir = validation_results_dir(sid, runs_root=runs_root)
    summary_file = results_dir / validation_result_summary_filename_for_account(account_id)
    jsonl_file = results_dir / validation_result_jsonl_filename_for_account(account_id)

    assert result_path == summary_file
    assert summary_file.exists()
    assert not jsonl_file.exists()

    entries = [child.name for child in results_dir.iterdir()]
    assert entries == [summary_file.name]

    stored_payload = json.loads(summary_file.read_text(encoding="utf-8"))
    assert stored_payload["decisions"] == [
        {
            "field_id": expected_ids[0],
            "decision": "strong",
            "rationale": "values diverge",
            "citations": [],
        },
        {
            "field_id": expected_ids[1],
            "decision": "no_case",
            "rationale": "matches reporting",
            "citations": [],
        },
    ]
