from __future__ import annotations

import json
import os
import sys
import types
from pathlib import Path
from typing import Any, Mapping

sys.modules.setdefault(
    "requests", types.SimpleNamespace(post=lambda *args, **kwargs: None)
)

from backend.ai.validation_builder import (
    ValidationPackWriter,
    build_validation_pack_for_account,
)
from backend.ai.validation_index import ValidationIndexEntry, ValidationPackIndexWriter
from backend.core.ai.paths import (
    validation_base_dir,
    validation_index_path,
    validation_pack_filename_for_account,
    validation_packs_dir,
    validation_result_jsonl_filename_for_account,
    validation_result_filename_for_account,
    validation_results_dir,
)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_json(path: Path) -> Mapping[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_validation_pack_path_generation(tmp_path: Path) -> None:
    sid = "SID001"
    runs_root = tmp_path / "runs_root"

    packs_dir = validation_packs_dir(sid, runs_root=runs_root)
    results_dir = validation_results_dir(sid, runs_root=runs_root)
    index_path = validation_index_path(sid, runs_root=runs_root)
    base_dir = validation_base_dir(sid, runs_root=runs_root)

    expected_base = (runs_root / sid / "ai_packs" / "validation").resolve()
    assert base_dir == expected_base
    assert packs_dir == (expected_base / "packs").resolve()
    assert results_dir == (expected_base / "results").resolve()
    assert index_path == (expected_base / "index.json").resolve()

    assert packs_dir.is_dir()
    assert results_dir.is_dir()
    assert index_path.parent.is_dir()

    assert validation_pack_filename_for_account(3) == "val_acc_003.jsonl"
    assert validation_pack_filename_for_account("12") == "val_acc_012.jsonl"
    assert validation_result_filename_for_account(7) == "acc_007.result.json"


def test_builds_pack_with_two_weak_fields(tmp_path: Path) -> None:
    sid = "SID002"
    runs_root = tmp_path / "runs"
    account_id = 12

    summary_payload = {
        "validation_requirements": {
            "requirements": [
                {
                    "field": "balance_owed",
                    "category": "activity",
                    "strength": "weak",
                    "documents": ["statement"],
                    "ai_needed": True,
                },
                {
                    "field": "creditor_remarks",
                    "category": "status",
                    "strength": "soft",
                    "ai_needed": True,
                    "notes": "recent remark change",
                    "conditional_gate": True,
                },
            ],
            "findings": [
                {"field": "balance_owed", "send_to_ai": True},
                {"field": "creditor_remarks", "send_to_ai": True},
            ],
            "field_consistency": {
                "balance_owed": {
                    "consensus": "split",
                    "disagreeing_bureaus": ["experian"],
                    "missing_bureaus": ["equifax"],
                    "history": {
                        "2y": {"late_payments": 1},
                        "7y": {"late_payments": 2},
                    },
                    "raw": {
                        "transunion": "$100",
                        "experian": "$150",
                    },
                    "normalized": {
                        "transunion": 100,
                        "experian": 150,
                    },
                },
                "creditor_remarks": {
                    "consensus": "majority",
                    "raw": {
                        "transunion": "Account closed by lender",
                        "experian": "Account closed by lender",
                        "equifax": "Account closed by creditor",
                    },
                    "normalized": {
                        "transunion": "account closed by lender",
                        "experian": "account closed by lender",
                        "equifax": "account closed by creditor",
                    },
                },
            },
        }
    }

    bureaus_payload = {
        "transunion": {
            "balance_owed": "$100",
            "creditor_remarks": "Account closed by lender",
        },
        "experian": {
            "balance_owed": "$150",
            "creditor_remarks": "Account closed by lender",
        },
        "equifax": {
            "balance_owed": None,
            "creditor_remarks": "Account closed by creditor",
        },
    }

    account_dir = runs_root / sid / "cases" / "accounts" / str(account_id)
    _write_json(account_dir / "summary.json", summary_payload)
    _write_json(account_dir / "bureaus.json", bureaus_payload)

    writer = ValidationPackWriter(sid, runs_root=runs_root)
    lines = writer.write_pack_for_account(account_id)

    assert len(lines) == 2
    fields = {line.payload["field"] for line in lines}
    assert fields == {"balance_owed", "creditor_remarks"}

    conditional_guidance = {
        line.payload["field"]: line.payload["prompt"]["guidance"] for line in lines
    }
    assert "Treat this as conditional" not in conditional_guidance["balance_owed"]
    assert "Treat this as conditional" in conditional_guidance["creditor_remarks"]

    pack_path = validation_packs_dir(sid, runs_root=runs_root) / validation_pack_filename_for_account(
        account_id
    )
    on_disk = _read_jsonl(pack_path)
    assert len(on_disk) == 2

    index_payload = _read_json(validation_index_path(sid, runs_root=runs_root))
    packs = index_payload.get("packs", [])
    assert len(packs) == 1
    assert packs[0]["weak_fields"] == ["balance_owed", "creditor_remarks"]
    assert packs[0]["lines"] == 2


def test_removed_fields_are_never_emitted(tmp_path: Path) -> None:
    sid = "SID099"
    runs_root = tmp_path / "runs"
    account_id = 5

    summary_payload = {
        "validation_requirements": {
            "requirements": [
                {
                    "field": "balance_owed",
                    "category": "activity",
                    "strength": "weak",
                    "ai_needed": True,
                },
                {
                    "field": "account_description",
                    "category": "status",
                    "strength": "weak",
                    "ai_needed": True,
                },
                {
                    "field": "dispute_status",
                    "category": "status",
                    "strength": "weak",
                    "ai_needed": True,
                },
                {
                    "field": "last_verified",
                    "category": "status",
                    "strength": "weak",
                    "ai_needed": True,
                },
            ],
            "findings": [
                {"field": "balance_owed", "send_to_ai": True},
                {"field": "account_description", "send_to_ai": True},
                {"field": "dispute_status", "send_to_ai": True},
                {"field": "last_verified", "send_to_ai": True},
            ],
            "field_consistency": {},
        }
    }

    bureaus_payload = {
        "transunion": {"balance_owed": "$200", "account_description": "Individual"},
        "experian": {"balance_owed": "$220", "dispute_status": "Not disputed"},
        "equifax": {"balance_owed": "$210", "last_verified": "2023-10-10"},
    }

    account_dir = runs_root / sid / "cases" / "accounts" / str(account_id)
    _write_json(account_dir / "summary.json", summary_payload)
    _write_json(account_dir / "bureaus.json", bureaus_payload)

    writer = ValidationPackWriter(sid, runs_root=runs_root)
    lines = writer.write_pack_for_account(account_id)

    assert len(lines) == 1
    assert lines[0].payload["field"] == "balance_owed"

    pack_path = validation_packs_dir(sid, runs_root=runs_root) / validation_pack_filename_for_account(
        account_id
    )
    on_disk = _read_jsonl(pack_path)
    assert {entry["field"] for entry in on_disk} == {"balance_owed"}

    index_payload = _read_json(validation_index_path(sid, runs_root=runs_root))
    packs = index_payload.get("packs", [])
    assert len(packs) == 1
    assert packs[0]["weak_fields"] == ["balance_owed"]
    assert packs[0]["lines"] == 1

def test_validation_index_round_trip(tmp_path: Path) -> None:
    sid = "SID003"
    runs_root = tmp_path / "runs"

    packs_dir = validation_packs_dir(sid, runs_root=runs_root)
    results_dir = validation_results_dir(sid, runs_root=runs_root)
    index_path = validation_index_path(sid, runs_root=runs_root)
    writer = ValidationPackIndexWriter(
        sid=sid,
        index_path=index_path,
        packs_dir=packs_dir,
        results_dir=results_dir,
    )

    pack_path1 = packs_dir / validation_pack_filename_for_account(1)
    summary_path1 = results_dir / validation_result_filename_for_account(1)
    jsonl_path1 = results_dir / validation_result_jsonl_filename_for_account(1)
    entry1 = ValidationIndexEntry(
        account_id=1,
        pack_path=pack_path1,
        result_jsonl_path=jsonl_path1,
        result_json_path=summary_path1,
        weak_fields=("balance_owed",),
        line_count=1,
        status="built",
    )
    pack_path2 = packs_dir / validation_pack_filename_for_account(2)
    summary_path2 = results_dir / validation_result_filename_for_account(2)
    jsonl_path2 = results_dir / validation_result_jsonl_filename_for_account(2)
    entry2 = ValidationIndexEntry(
        account_id=2,
        pack_path=pack_path2,
        result_jsonl_path=jsonl_path2,
        result_json_path=summary_path2,
        weak_fields=("creditor_remarks", "balance_owed"),
        line_count=2,
        status="built",
    )

    writer.bulk_upsert([entry1, entry2])

    accounts = writer.load_accounts()
    assert set(accounts) == {1, 2}
    assert accounts[1]["weak_fields"] == ["balance_owed"]
    assert accounts[2]["weak_fields"] == ["creditor_remarks", "balance_owed"]
    assert accounts[2]["lines"] == 2


def test_manifest_updated_after_first_pack(tmp_path: Path, monkeypatch) -> None:
    sid = "SID004"
    runs_root = tmp_path / "runs"
    account_id = 5

    monkeypatch.setenv("RUNS_ROOT", str(runs_root))

    summary_payload = {
        "validation_requirements": {
            "requirements": [
                {
                    "field": "balance_owed",
                    "strength": "weak",
                    "ai_needed": True,
                }
            ],
            "findings": [
                {"field": "balance_owed", "send_to_ai": True}
            ],
            "field_consistency": {},
        }
    }

    account_dir = runs_root / sid / "cases" / "accounts" / str(account_id)
    summary_path = account_dir / "summary.json"
    bureaus_path = account_dir / "bureaus.json"
    _write_json(summary_path, summary_payload)
    _write_json(bureaus_path, {})

    lines = build_validation_pack_for_account(sid, account_id, summary_path, bureaus_path)
    assert len(lines) == 1

    manifest_path = runs_root / sid / "manifest.json"
    manifest_payload = _read_json(manifest_path)

    packs_validation = manifest_payload["ai"]["packs"]["validation"]
    expected_packs_dir = str(validation_packs_dir(sid, runs_root=runs_root))
    expected_results_dir = str(validation_results_dir(sid, runs_root=runs_root))
    expected_index = str(validation_index_path(sid, runs_root=runs_root))

    assert packs_validation["packs_dir"] == expected_packs_dir
    assert packs_validation["results_dir"] == expected_results_dir
    assert packs_validation["index"] == expected_index

    ai_validation = manifest_payload["ai"]["validation"]
    expected_base = str(validation_base_dir(sid, runs_root=runs_root))
    assert ai_validation["accounts_dir"] == expected_base


def test_build_validation_pack_idempotent(tmp_path: Path, monkeypatch) -> None:
    sid = "SID005"
    runs_root = tmp_path / "runs"
    account_id = 8

    monkeypatch.setenv("RUNS_ROOT", str(runs_root))

    summary_payload = {
        "validation_requirements": {
            "requirements": [
                {
                    "field": "balance_owed",
                    "strength": "weak",
                    "ai_needed": True,
                },
                {
                    "field": "account_status",
                    "strength": "weak",
                    "ai_needed": True,
                },
            ],
            "findings": [
                {"field": "balance_owed", "send_to_ai": True},
                {"field": "account_status", "send_to_ai": True},
            ],
            "field_consistency": {},
        }
    }

    account_dir = runs_root / sid / "cases" / "accounts" / str(account_id)
    summary_path = account_dir / "summary.json"
    bureaus_path = account_dir / "bureaus.json"
    _write_json(summary_path, summary_payload)
    _write_json(bureaus_path, {})

    first_lines = build_validation_pack_for_account(sid, account_id, summary_path, bureaus_path)
    second_lines = build_validation_pack_for_account(sid, account_id, summary_path, bureaus_path)

    assert len(first_lines) == len(second_lines) == 2

    pack_path = validation_packs_dir(sid, runs_root=runs_root) / validation_pack_filename_for_account(
        account_id
    )
    on_disk = _read_jsonl(pack_path)
    assert len(on_disk) == 2

    index_payload = _read_json(validation_index_path(sid, runs_root=runs_root))
    packs = index_payload.get("packs", [])
    assert len(packs) == 1
    assert packs[0]["lines"] == 2
    assert packs[0]["weak_fields"] == ["balance_owed", "account_status"]

    results_path = validation_results_dir(sid, runs_root=runs_root)
    result_file = results_path / validation_result_filename_for_account(account_id)
    assert result_file.parent == results_path
    assert not result_file.exists()
