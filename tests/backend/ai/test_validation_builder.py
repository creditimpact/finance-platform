import json
from pathlib import Path

import pytest

from backend.ai.validation_builder import ValidationPackWriter
from backend.core.ai.paths import (
    validation_pack_filename_for_account,
    validation_packs_dir,
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


def test_no_pack_for_ineligible_findings(tmp_path: Path) -> None:
    sid = "SID100"
    account_id = 1
    runs_root = tmp_path / "runs"

    summary_payload = _build_summary(
        {
            "field": "payment_status",
            "is_mismatch": True,
            "ai_needed": True,
            "send_to_ai": True,
        }
    )

    account_dir = runs_root / sid / "cases" / "accounts" / str(account_id)
    _write_json(account_dir / "summary.json", summary_payload)

    writer = ValidationPackWriter(sid, runs_root=runs_root)
    lines = writer.write_pack_for_account(account_id)

    assert lines == []

    pack_path = (
        validation_packs_dir(sid, runs_root=runs_root)
        / validation_pack_filename_for_account(account_id)
    )
    assert not pack_path.exists()
    assert writer._index_writer.load_accounts() == {}


def test_account_type_mismatch_produces_single_line(tmp_path: Path) -> None:
    sid = "SID200"
    account_id = 2
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

    writer = ValidationPackWriter(sid, runs_root=runs_root)
    lines = writer.write_pack_for_account(account_id)

    assert len(lines) == 1
    payload = lines[0].payload
    assert payload["field"] == "account_type"
    assert payload["finding"] == finding

    pack_path = (
        validation_packs_dir(sid, runs_root=runs_root)
        / validation_pack_filename_for_account(account_id)
    )
    assert pack_path.exists()

    stored = writer._index_writer.load_accounts()
    assert account_id in stored
    stored_entry = stored[account_id]
    assert stored_entry["account_id"] == account_id
    expected_pack_rel = f"packs/{validation_pack_filename_for_account(account_id)}"
    assert stored_entry["pack"] == expected_pack_rel
    assert stored_entry["weak_fields"] == ["account_type"]
    assert stored_entry["lines"] == 1
    assert stored_entry["status"] == "built"


def test_two_year_history_toggle_enabled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sid = "SID300"
    account_id = 3
    runs_root = tmp_path / "runs"

    monkeypatch.setenv("VALIDATION_ALLOW_HISTORY_2Y_AI", "1")

    finding = {
        "field": "two_year_payment_history",
        "is_mismatch": True,
        "ai_needed": False,
        "send_to_ai": True,
    }

    account_dir = runs_root / sid / "cases" / "accounts" / str(account_id)
    _write_json(account_dir / "summary.json", _build_summary(finding))

    writer = ValidationPackWriter(sid, runs_root=runs_root)
    lines = writer.write_pack_for_account(account_id)

    assert len(lines) == 1
    assert lines[0].payload["field"] == "two_year_payment_history"


def test_two_year_history_toggle_disabled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sid = "SID301"
    account_id = 31
    runs_root = tmp_path / "runs"

    monkeypatch.setenv("VALIDATION_ALLOW_HISTORY_2Y_AI", "0")

    finding = {
        "field": "two_year_payment_history",
        "is_mismatch": True,
        "ai_needed": False,
        "send_to_ai": True,
    }

    account_dir = runs_root / sid / "cases" / "accounts" / str(account_id)
    _write_json(account_dir / "summary.json", _build_summary(finding))

    writer = ValidationPackWriter(sid, runs_root=runs_root)
    lines = writer.write_pack_for_account(account_id)

    assert lines == []


def test_two_year_history_requires_send_flag(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sid = "SID302"
    account_id = 32
    runs_root = tmp_path / "runs"

    monkeypatch.setenv("VALIDATION_ALLOW_HISTORY_2Y_AI", "1")

    finding = {
        "field": "two_year_payment_history",
        "is_mismatch": True,
        "ai_needed": False,
        "send_to_ai": False,
    }

    account_dir = runs_root / sid / "cases" / "accounts" / str(account_id)
    _write_json(account_dir / "summary.json", _build_summary(finding))

    writer = ValidationPackWriter(sid, runs_root=runs_root)
    lines = writer.write_pack_for_account(account_id)

    assert lines == []


def test_excluded_fields_never_generate_packs(tmp_path: Path) -> None:
    sid = "SID400"
    account_id = 4
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

    pack_path = (
        validation_packs_dir(sid, runs_root=runs_root)
        / validation_pack_filename_for_account(account_id)
    )
    assert not pack_path.exists()
