from __future__ import annotations

import json
import sys
import logging
import types
from pathlib import Path
from typing import Any, Mapping

import pytest

sys.modules.setdefault(
    "requests", types.SimpleNamespace(post=lambda *args, **kwargs: None)
)

from backend.ai.validation_builder import (
    ValidationPackWriter,
    build_validation_pack_for_account,
    build_validation_packs_for_run,
)
from backend.ai.validation_results import (
    mark_validation_pack_sent,
    store_validation_result,
)
from backend.core.ai.paths import (
    validation_index_path,
    validation_pack_filename_for_account,
    validation_packs_dir,
    validation_result_jsonl_filename_for_account,
    validation_result_filename_for_account,
    validation_results_dir,
    validation_logs_path,
)
from backend.validation.manifest import rewrite_index_to_canonical_layout


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _read_index(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _index_entry_for_account(index_payload: Mapping[str, Any], account_id: int) -> Mapping[str, Any]:
    packs = index_payload.get("packs", [])
    for entry in packs:
        if isinstance(entry, Mapping) and entry.get("account_id") == account_id:
            return entry
    raise AssertionError(f"account {account_id} not found in index")


def test_writer_builds_pack_lines(tmp_path: Path) -> None:
    sid = "S123"
    runs_root = tmp_path / "runs"

    summary_payload = {
        "validation_requirements": {
            "findings": [
                {
                    "field": "account_rating",
                    "category": "status",
                    "documents": ["statement", "   "],
                    "strength": "weak",
                    "notes": "rating disagreement",
                    "is_mismatch": True,
                    "ai_needed": True,
                    "send_to_ai": True,
                },
                {
                    "field": "account_status",
                    "category": "status",
                    "ai_needed": False,
                    "send_to_ai": False,
                },
            ],
            "field_consistency": {
                "account_rating": {
                    "consensus": "split",
                    "disagreeing_bureaus": ["experian"],
                    "missing_bureaus": ["equifax"],
                    "history": {"2y": {"late_payments": 1}},
                    "raw": {
                        "transunion": "1",
                        "experian": "2",
                    },
                    "normalized": {
                        "transunion": "1",
                        "experian": "2",
                    },
                }
            },
        }
    }

    bureaus_payload = {
        "transunion": {"account_rating": "1"},
        "experian": {"account_rating": "2"},
        "equifax": {"account_rating": None},
    }

    account_dir = runs_root / sid / "cases" / "accounts" / "1"
    _write_json(account_dir / "summary.json", summary_payload)
    _write_json(account_dir / "bureaus.json", bureaus_payload)

    writer = ValidationPackWriter(sid, runs_root=runs_root)
    result = writer.write_all_packs()

    assert set(result) == {1}
    lines = result[1]
    assert len(lines) >= 1

    payload = None
    for line in lines:
        if line.payload.get("field") == "account_rating":
            payload = line.payload
            break

    assert payload is not None
    assert payload["id"] == "acc_001__account_rating"
    assert payload["sid"] == sid
    assert payload["account_id"] == 1
    assert payload["field"] == "account_rating"

    finding = payload["finding"]
    assert finding["field"] == "account_rating"
    assert finding["strength"] == "weak"
    assert finding.get("category") == "status"
    assert finding.get("documents") == ["statement", "   "]
    assert finding.get("notes") == "rating disagreement"
    assert finding.get("bureau_values", {}).get("transunion", {}).get("raw") == "1"

    prompt = payload["prompt"]
    assert isinstance(prompt, dict)
    assert prompt["system"].startswith(
        "You are a credit dispute adjudication assistant."
    )
    assert prompt["user"].startswith(
        "You are given a single field finding extracted"
    )
    assert "<finding blob here>" not in prompt["user"]
    assert payload["finding_json"] in prompt["user"]

    expected_output = payload["expected_output"]
    assert expected_output["properties"]["decision"]["enum"] == [
        "strong_actionable",
        "supportive_needs_companion",
        "neutral_context_only",
        "no_case",
    ]
    assert expected_output["properties"]["citations"]["minItems"] == 1
    assert set(
        expected_output["properties"]["checks"]["required"]
    ) == {
        "materiality",
        "supports_consumer",
        "doc_requirements_met",
        "mismatch_code",
    }

    pack_path = runs_root / sid / "ai_packs" / "validation" / "packs" / "val_acc_001.jsonl"
    on_disk = _read_jsonl(pack_path)
    assert any(line["id"] == payload["id"] for line in on_disk)


def test_writer_skips_account_number_display(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    sid = "S890"
    runs_root = tmp_path / "runs"

    summary_payload = {
        "validation_requirements": {
            "findings": [
                {
                    "field": "account_number_display",
                    "category": "open_ident",
                    "strength": "weak",
                    "documents": ["statement"],
                    "is_mismatch": True,
                    "ai_needed": True,
                    "send_to_ai": True,
                }
            ],
            "field_consistency": {
                "account_number_display": {
                    "raw": {"transunion": "123456789012"},
                    "normalized": {
                        "transunion": {
                            "display": "1234-5678-9012",
                            "last4": "9012",
                        },
                        "experian": {"display": "9012", "last4": "9012"},
                    },
                }
            },
        }
    }

    bureaus_payload = {
        "transunion": {"account_number_display": "123456789012"},
        "experian": {"account_number_display": "Account 9012"},
        "equifax": {},
    }

    account_dir = runs_root / sid / "cases" / "accounts" / "5"
    _write_json(account_dir / "summary.json", summary_payload)
    _write_json(account_dir / "bureaus.json", bureaus_payload)

    writer = ValidationPackWriter(sid, runs_root=runs_root)
    caplog.set_level(logging.INFO, logger="backend.ai.validation_builder")
    lines = writer.write_pack_for_account(5)

    assert lines == []

    pack_path = runs_root / sid / "ai_packs" / "validation" / "packs" / "val_acc_005.jsonl"
    assert not pack_path.exists()

    messages = [record.getMessage() for record in caplog.records]
    assert (
        "validation pack skipped: no eligible lines (sid=S890 account=005)" in messages
    )

    index_path = validation_index_path(sid, runs_root=runs_root)
    assert not index_path.exists()
def test_writer_uses_bureau_fallback(tmp_path: Path) -> None:
    sid = "S234"
    runs_root = tmp_path / "runs"

    summary_payload = {
        "validation_requirements": {
            "findings": [
                {
                    "field": "account_type",
                    "category": "open_ident",
                    "strength": "soft",
                    "is_mismatch": True,
                    "ai_needed": True,
                    "notes": "type mismatch",
                    "send_to_ai": True,
                }
            ],
            "field_consistency": {},
        }
    }
    bureaus_payload = {
        "transunion": {"account_type": "Mortgage"},
    }

    account_dir = runs_root / sid / "cases" / "accounts" / "7"
    _write_json(account_dir / "summary.json", summary_payload)
    _write_json(account_dir / "bureaus.json", bureaus_payload)

    writer = ValidationPackWriter(sid, runs_root=runs_root)
    lines = writer.write_pack_for_account(7)

    assert len(lines) == 1
    payload = lines[0].payload
    finding = payload["finding"]
    assert finding["notes"] == "type mismatch"
    assert finding["bureau_values"]["transunion"]["raw"] == "Mortgage"

    pack_path = runs_root / sid / "ai_packs" / "validation" / "packs" / "val_acc_007.jsonl"
    on_disk = _read_jsonl(pack_path)
    assert on_disk[0]["finding"]["bureau_values"]["transunion"]["raw"] == "Mortgage"


def test_writer_skips_strong_fields(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    sid = "S345"
    runs_root = tmp_path / "runs"

    summary_payload = {
        "validation_requirements": {
            "findings": [
                {
                    "field": "payment_status",
                    "category": "status",
                    "strength": "strong",
                    "is_mismatch": True,
                    "ai_needed": True,
                    "send_to_ai": True,
                }
            ],
            "field_consistency": {},
        }
    }

    account_dir = runs_root / sid / "cases" / "accounts" / "9"
    _write_json(account_dir / "summary.json", summary_payload)

    writer = ValidationPackWriter(sid, runs_root=runs_root)
    caplog.set_level(logging.INFO, logger="backend.ai.validation_builder")
    lines = writer.write_pack_for_account(9)

    assert lines == []

    pack_path = runs_root / sid / "ai_packs" / "validation" / "packs" / "val_acc_009.jsonl"
    assert not pack_path.exists()

    messages = [record.getMessage() for record in caplog.records]
    assert (
        "validation pack skipped: no eligible lines (sid=S345 account=009)" in messages
    )

    index_path = validation_index_path(sid, runs_root=runs_root)
    assert not index_path.exists()


def test_writer_updates_index(tmp_path: Path) -> None:
    sid = "S456"
    runs_root = tmp_path / "runs"

    summary_payload = {
        "validation_requirements": {
            "findings": [
                {
                    "field": "account_type",
                    "category": "open_ident",
                    "strength": "weak",
                    "is_mismatch": True,
                    "ai_needed": True,
                    "send_to_ai": True,
                },
                {
                    "field": "account_rating",
                    "category": "status",
                    "strength": "weak",
                    "is_mismatch": True,
                    "ai_needed": True,
                    "send_to_ai": True,
                },
            ],
            "field_consistency": {
                "account_type": {
                    "raw": {"transunion": "Mortgage"},
                },
                "account_rating": {
                    "raw": {"transunion": "1"},
                },
            },
        }
    }

    account_dir = runs_root / sid / "cases" / "accounts" / "1"
    _write_json(account_dir / "summary.json", summary_payload)

    writer = ValidationPackWriter(sid, runs_root=runs_root)
    writer.write_all_packs()

    index_path = validation_index_path(sid, runs_root=runs_root)
    index_payload = _read_index(index_path)

    assert index_payload["schema_version"] == 2
    assert index_payload["sid"] == sid
    assert index_payload["root"] == "."
    assert index_payload["packs_dir"] == "packs"
    assert index_payload["results_dir"] == "results"
    assert len(index_payload["packs"]) == 1

    entry = index_payload["packs"][0]
    assert entry["account_id"] == 1
    assert entry["pack"] == "packs/val_acc_001.jsonl"
    assert "result_json" not in entry
    assert "result_jsonl" not in entry
    assert entry["lines"] == 2
    assert entry["weak_fields"] == ["account_type", "account_rating"]
    assert entry["status"] == "built"
    assert isinstance(entry["built_at"], str) and entry["built_at"].endswith("Z")
    assert isinstance(entry.get("source_hash"), str) and len(entry["source_hash"]) == 64

    # Update summary to remove one requirement and rebuild.
    summary_payload["validation_requirements"]["findings"] = [
        summary_payload["validation_requirements"]["findings"][0]
    ]
    _write_json(account_dir / "summary.json", summary_payload)

    writer.write_pack_for_account(1)

    refreshed_index = _read_index(index_path)
    assert len(refreshed_index["packs"]) == 1
    refreshed_entry = refreshed_index["packs"][0]
    assert refreshed_entry["lines"] == 1
    assert refreshed_entry["weak_fields"] == ["account_type"]
    assert isinstance(refreshed_entry.get("source_hash"), str)
    assert refreshed_entry["source_hash"] != entry["source_hash"]


def _seed_validation_account(
    runs_root: Path,
    sid: str,
    account_id: int,
    *,
    field: str = "account_rating",
) -> None:
    account_dir = runs_root / sid / "cases" / "accounts" / str(account_id)
    summary_payload = {
        "validation_requirements": {
            "findings": [
                {
                    "field": field,
                    "category": "activity",
                    "strength": "weak",
                    "is_mismatch": True,
                    "ai_needed": True,
                    "send_to_ai": True,
                }
            ],
            "field_consistency": {
                field: {"raw": {"transunion": "1"}},
            },
        }
    }
    bureaus_payload = {
        "transunion": {field: "1"},
        "experian": {field: "2"},
    }
    _write_json(account_dir / "summary.json", summary_payload)
    _write_json(account_dir / "bureaus.json", bureaus_payload)


def test_writer_deduplicates_duplicate_findings(tmp_path: Path) -> None:
    sid = "S571"
    runs_root = tmp_path / "runs"
    account_id = 6

    summary_payload = {
        "validation_requirements": {
            "findings": [
                {
                    "field": "account_type",
                    "category": "open_ident",
                    "strength": "weak",
                    "is_mismatch": True,
                    "ai_needed": True,
                    "send_to_ai": True,
                },
                {
                    "field": "account_type",
                    "category": "open_ident",
                    "strength": "weak",
                    "is_mismatch": True,
                    "ai_needed": True,
                    "send_to_ai": True,
                },
            ],
            "field_consistency": {
                "account_type": {"raw": {"transunion": "Mortgage"}},
            },
        }
    }
    bureaus_payload = {
        "transunion": {"account_type": "Mortgage"},
        "experian": {"account_type": "Installment"},
    }

    account_dir = runs_root / sid / "cases" / "accounts" / str(account_id)
    _write_json(account_dir / "summary.json", summary_payload)
    _write_json(account_dir / "bureaus.json", bureaus_payload)

    writer = ValidationPackWriter(sid, runs_root=runs_root)

    first_lines = writer.write_pack_for_account(account_id)
    assert len(first_lines) == 1
    first_payload = first_lines[0].payload
    assert first_payload["field"] == "account_type"

    second_lines = writer.write_pack_for_account(account_id)
    assert len(second_lines) == 1
    assert second_lines[0].payload["id"] == first_payload["id"]

    pack_path = (
        validation_packs_dir(sid, runs_root=runs_root)
        / validation_pack_filename_for_account(account_id)
    )
    on_disk = _read_jsonl(pack_path)
    assert len(on_disk) == 1


def test_mark_validation_pack_sent_updates_index(tmp_path: Path) -> None:
    sid = "S567"
    runs_root = tmp_path / "runs"
    account_id = 3

    _seed_validation_account(runs_root, sid, account_id)

    writer = ValidationPackWriter(sid, runs_root=runs_root)
    lines = writer.write_pack_for_account(account_id)
    assert len(lines) == 1

    index_path = validation_index_path(sid, runs_root=runs_root)
    index_payload = _read_index(index_path)
    entry = _index_entry_for_account(index_payload, account_id)
    assert entry["status"] == "built"
    assert isinstance(entry.get("source_hash"), str)

    mark_validation_pack_sent(
        sid,
        account_id,
        runs_root=runs_root,
        request_lines=len(lines),
        model="gpt-test",
    )

    updated_payload = _read_index(index_path)
    updated_entry = _index_entry_for_account(updated_payload, account_id)
    assert updated_entry["status"] == "sent"
    assert updated_entry["request_lines"] == len(lines)
    assert updated_entry["model"] == "gpt-test"
    assert "completed_at" not in updated_entry
    assert isinstance(updated_entry["sent_at"], str)
    assert updated_entry["source_hash"] == entry["source_hash"]


def test_store_validation_result_updates_index_and_writes_file(
    tmp_path: Path,
) -> None:
    sid = "S678"
    runs_root = tmp_path / "runs"
    account_id = 4

    _seed_validation_account(runs_root, sid, account_id)

    writer = ValidationPackWriter(sid, runs_root=runs_root)
    lines = writer.write_pack_for_account(account_id)
    mark_validation_pack_sent(
        sid,
        account_id,
        runs_root=runs_root,
        request_lines=len(lines),
        model="gpt-infer",
    )

    response_payload = {
        "decision_per_field": [
            {
                "field": "account_rating",
                "decision": "strong",
                "rationale": "values diverge",
                "confidence": 0.81,
            }
        ],
        "raw_response": {"id": "resp_123"},
    }

    result_path = store_validation_result(
        sid,
        account_id,
        response_payload,
        runs_root=runs_root,
        status="done",
        request_lines=len(lines),
        model="gpt-infer",
    )

    stored_lines = [json.loads(line) for line in result_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert stored_lines
    first_line = stored_lines[0]
    assert first_line["account_id"] == account_id
    assert first_line["decision"] == "strong"
    assert first_line["rationale"] == "values diverge"
    assert first_line["citations"] == []
    assert first_line["legacy_decision"] == "strong"

    results_dir = validation_results_dir(sid, runs_root=runs_root)
    jsonl_path = (
        results_dir / validation_result_jsonl_filename_for_account(account_id)
    )
    assert jsonl_path.exists()

    index_path = validation_index_path(sid, runs_root=runs_root)
    index_entry = _index_entry_for_account(_read_index(index_path), account_id)
    assert index_entry["status"] == "completed"
    assert index_entry["result_json"] == "results/acc_004.result.jsonl"
    assert index_entry["lines"] == len(lines)
    assert "error" not in index_entry


def test_reason_metadata_flag_is_ignored(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VALIDATION_REASON_ENABLED", "1")

    sid = "S901"
    runs_root = tmp_path / "runs"
    account_id = 9

    _seed_validation_account(runs_root, sid, account_id)

    writer = ValidationPackWriter(sid, runs_root=runs_root)
    lines = writer.write_pack_for_account(account_id)

    assert len(lines) == 1
    pack_payload = lines[0].payload
    assert "reason" not in pack_payload

    pack_path = runs_root / sid / "ai_packs" / "validation" / "packs" / "val_acc_009.jsonl"
    on_disk = _read_jsonl(pack_path)
    assert "reason" not in on_disk[0]

def test_store_validation_result_error(tmp_path: Path) -> None:
    sid = "S789"
    runs_root = tmp_path / "runs"
    account_id = 5

    _seed_validation_account(runs_root, sid, account_id, field="account_type")

    writer = ValidationPackWriter(sid, runs_root=runs_root)
    writer.write_pack_for_account(account_id)

    result_path = store_validation_result(
        sid,
        account_id,
        {"raw_response": {"error": "timeout"}},
        runs_root=runs_root,
        status="error",
        error="api_timeout",
    )

    stored_text = result_path.read_text(encoding="utf-8")
    assert stored_text.strip() == ""

    index_path = validation_index_path(sid, runs_root=runs_root)
    entry = _index_entry_for_account(_read_index(index_path), account_id)
    assert entry["status"] == "failed"
    assert entry["error"] == "api_timeout"
    assert isinstance(entry["completed_at"], str)
    assert isinstance(entry.get("source_hash"), str)
    assert "result_json" not in entry


def test_writer_appends_log_entries(tmp_path: Path) -> None:
    sid = "S910"
    runs_root = tmp_path / "runs"

    _seed_validation_account(runs_root, sid, 8)

    writer = ValidationPackWriter(sid, runs_root=runs_root)
    lines = writer.write_pack_for_account(8)

    assert len(lines) == 1

    log_path = runs_root / sid / "ai_packs" / "validation" / "logs.txt"
    assert log_path.exists()

    log_entries = [
        json.loads(line)
        for line in log_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert len(log_entries) == 1
    entry = log_entries[0]
    assert entry["account_index"] == 8
    assert entry["weak_count"] == 1
    assert entry["statuses"] == ["pack_written"]
    assert entry["mode"] == "per_account"


def test_writer_logs_pack_metrics(tmp_path: Path) -> None:
    sid = "S611"
    runs_root = tmp_path / "runs"
    account_id = 11

    _seed_validation_account(runs_root, sid, account_id)

    writer = ValidationPackWriter(sid, runs_root=runs_root)
    lines = writer.write_pack_for_account(account_id)

    assert len(lines) == 1

    logs_path = validation_logs_path(sid, runs_root=runs_root)
    log_entries = _read_jsonl(logs_path)
    assert log_entries, "expected log entries to be written"
    entry = log_entries[-1]

    assert entry["pack_size_bytes"] > 0
    assert entry["pack_size_kb"] > 0
    assert entry["cumulative_size"]["count"] >= 1
    assert entry["cumulative_size"]["max_bytes"] >= entry["pack_size_bytes"]
    assert entry["fields_emitted"] == ["account_rating"]
    assert entry["cumulative_field_counts"]["account_rating"] >= 1


def test_writer_blocks_pack_exceeding_size_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sid = "S612"
    runs_root = tmp_path / "runs"
    account_id = 12

    summary_payload = {
        "validation_requirements": {
            "findings": [
                {
                    "field": "account_type",
                    "category": "open_ident",
                    "strength": "weak",
                    "is_mismatch": True,
                    "ai_needed": True,
                    "send_to_ai": True,
                    "notes": "X" * 4096,
                }
            ],
            "field_consistency": {
                "account_type": {
                    "raw": {"transunion": "Mortgage", "experian": "Installment"}
                }
            },
        }
    }
    bureaus_payload = {
        "transunion": {"account_type": "Mortgage"},
        "experian": {"account_type": "Installment"},
    }

    account_dir = runs_root / sid / "cases" / "accounts" / str(account_id)
    _write_json(account_dir / "summary.json", summary_payload)
    _write_json(account_dir / "bureaus.json", bureaus_payload)

    monkeypatch.setenv("VALIDATION_PACK_MAX_SIZE_KB", "1")

    writer = ValidationPackWriter(sid, runs_root=runs_root)
    lines = writer.write_pack_for_account(account_id)

    assert lines == []

    pack_path = (
        runs_root
        / sid
        / "ai_packs"
        / "validation"
        / "packs"
        / "val_acc_012.jsonl"
    )
    assert not pack_path.exists()

    index_path = validation_index_path(sid, runs_root=runs_root)
    assert not index_path.exists()

    log_entries = _read_jsonl(validation_logs_path(sid, runs_root=runs_root))
    assert log_entries, "expected blocked pack to be logged"
    entry = log_entries[-1]
    assert entry["statuses"] == ["pack_blocked_max_size"]
    assert entry["pack_size_limit_kb"] == 1.0
    assert entry["pack_size_bytes"] > 0
    assert entry["cumulative_size"]["count"] == 0


def test_build_validation_pack_respects_env_toggle(
    tmp_path: Path, monkeypatch
) -> None:
    sid = "S911"
    runs_root = tmp_path / "runs"

    account_dir = runs_root / sid / "cases" / "accounts" / "5"
    summary_path = account_dir / "summary.json"
    bureaus_path = account_dir / "bureaus.json"

    account_dir.mkdir(parents=True, exist_ok=True)
    summary_payload = {
        "validation_requirements": {
            "findings": [
                {
                    "field": "account_rating",
                    "category": "status",
                    "strength": "weak",
                    "is_mismatch": True,
                    "ai_needed": True,
                    "send_to_ai": True,
                }
            ],
            "field_consistency": {
                "account_rating": {"raw": {"transunion": "1"}}
            },
        }
    }
    bureaus_payload = {"transunion": {"account_rating": "1"}}
    _write_json(summary_path, summary_payload)
    _write_json(bureaus_path, bureaus_payload)

    monkeypatch.setenv("VALIDATION_PACKS_ENABLED", "0")

    lines = build_validation_pack_for_account(
        sid,
        5,
        summary_path,
        bureaus_path,
    )

    assert lines == []

    pack_path = runs_root / sid / "ai_packs" / "validation" / "packs" / "val_acc_005.jsonl"
    assert not pack_path.exists()


def test_build_validation_packs_for_run_does_not_auto_send(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sid = "S601"
    runs_root = tmp_path / "runs"
    account_dir = runs_root / sid / "cases" / "accounts" / "1"
    account_dir.mkdir(parents=True, exist_ok=True)

    summary_payload = {
        "validation_requirements": {
            "findings": [
                {
                    "field": "account_rating",
                    "category": "status",
                    "strength": "weak",
                    "is_mismatch": True,
                    "ai_needed": True,
                    "send_to_ai": True,
                }
            ],
            "field_consistency": {
                "account_rating": {"raw": {"transunion": "1"}}
            },
        }
    }
    bureaus_payload = {"transunion": {"account_rating": "1"}}

    _write_json(account_dir / "summary.json", summary_payload)
    _write_json(account_dir / "bureaus.json", bureaus_payload)

    monkeypatch.setenv("ENABLE_VALIDATION_SENDER", "1")
    monkeypatch.setenv("AUTO_VALIDATION_SEND", "1")
    monkeypatch.setenv("VALIDATION_SEND_ON_BUILD", "1")
    monkeypatch.setitem(
        sys.modules,
        "requests",
        types.SimpleNamespace(post=lambda *args, **kwargs: None),
    )

    captured: dict[str, Any] = {}

    def _fake_send(manifest: Any) -> list[dict[str, Any]]:  # pragma: no cover - safety
        captured["manifest"] = manifest
        return []

    monkeypatch.setattr(
        "backend.validation.send_packs.send_validation_packs",
        _fake_send,
    )

    build_validation_packs_for_run(sid, runs_root=runs_root)

    assert captured == {}


def test_rewrite_index_to_canonical_layout(tmp_path: Path) -> None:
    sid = "S777"
    runs_root = tmp_path / "runs"

    account_dir = runs_root / sid / "cases" / "accounts" / "1"
    summary_payload = {
        "validation_requirements": {
            "findings": [
                {
                    "field": "account_rating",
                    "category": "status",
                    "strength": "weak",
                    "is_mismatch": True,
                    "ai_needed": True,
                    "send_to_ai": True,
                }
            ],
            "field_consistency": {
                "account_rating": {
                    "raw": {"transunion": "1"},
                }
            },
        }
    }
    bureaus_payload = {
        "transunion": {"account_rating": "1"},
        "experian": {"account_rating": "2"},
    }

    _write_json(account_dir / "summary.json", summary_payload)
    _write_json(account_dir / "bureaus.json", bureaus_payload)

    writer = ValidationPackWriter(sid, runs_root=runs_root)
    writer.write_pack_for_account(1)

    index_path = validation_index_path(sid, runs_root=runs_root)
    payload = _read_index(index_path)
    entry = _index_entry_for_account(payload, 1)

    payload["packs_dir"] = "../cases/accounts"
    payload["results_dir"] = "../cases/accounts/results"
    entry["pack"] = "../cases/accounts/1/pack.jsonl"
    entry["result_json"] = "../cases/accounts/1/result.jsonl"
    _write_json(index_path, payload)

    _, changed = rewrite_index_to_canonical_layout(index_path, runs_root=runs_root)
    assert changed is True

    rewritten = _read_index(index_path)
    assert rewritten["packs_dir"] == "packs"
    assert rewritten["results_dir"] == "results"

    rewritten_entry = _index_entry_for_account(rewritten, 1)
    assert rewritten_entry["pack"] == "packs/val_acc_001.jsonl"
    assert rewritten_entry["result_json"] == "results/acc_001.result.jsonl"
    assert "result_jsonl" not in rewritten_entry

    _, unchanged = rewrite_index_to_canonical_layout(index_path, runs_root=runs_root)
    assert unchanged is False
