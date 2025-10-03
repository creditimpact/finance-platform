from __future__ import annotations

import json
import sys
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
    validation_result_jsonl_filename_for_account,
    validation_result_filename_for_account,
    validation_results_dir,
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
            "requirements": [
                {
                    "field": "balance_owed",
                    "category": "activity",
                    "min_days": 30,
                    "documents": ["statement", "   "],
                    "strength": "weak",
                    "ai_needed": True,
                },
                {
                    "field": "account_status",
                    "category": "status",
                    "ai_needed": False,
                },
            ],
            "field_consistency": {
                "balance_owed": {
                    "consensus": "split",
                    "disagreeing_bureaus": ["experian"],
                    "missing_bureaus": ["equifax"],
                    "history": {"2y": {"late_payments": 1}},
                    "raw": {
                        "transunion": "$100",
                        "experian": "$150",
                    },
                    "normalized": {
                        "transunion": 100,
                        "experian": 150,
                    },
                }
            },
        }
    }

    bureaus_payload = {
        "transunion": {"balance_owed": "$100"},
        "experian": {"balance_owed": "$155"},
        "equifax": {"balance_owed": None},
    }

    account_dir = runs_root / sid / "cases" / "accounts" / "1"
    _write_json(account_dir / "summary.json", summary_payload)
    _write_json(account_dir / "bureaus.json", bureaus_payload)

    writer = ValidationPackWriter(sid, runs_root=runs_root)
    result = writer.write_all_packs()

    assert set(result) == {1}
    lines = result[1]
    assert len(lines) == 1

    payload = lines[0].payload
    assert payload["id"] == "acc_001__balance_owed"
    assert payload["field"] == "balance_owed"
    assert payload["strength"] == "weak"
    assert payload["documents"] == ["statement"]

    bureaus = payload["bureaus"]
    assert bureaus["transunion"]["raw"] == "$100"
    assert bureaus["experian"]["normalized"] == 150
    assert bureaus["equifax"]["raw"] is None

    context = payload["context"]
    assert context["consensus"] == "split"
    assert context["disagreeing_bureaus"] == ["experian"]
    assert context["missing_bureaus"] == ["equifax"]
    assert context["history"] == {"2y": {"late_payments": 1}}

    prompt = payload["prompt"]
    assert prompt["user"]["field_key"] == "balance_owed"
    assert "strong" in prompt["guidance"].lower()

    expected_output = payload["expected_output"]
    assert expected_output["properties"]["decision"]["enum"] == ["strong", "no_case"]

    pack_path = runs_root / sid / "ai_packs" / "validation" / "packs" / "val_acc_001.jsonl"
    on_disk = _read_jsonl(pack_path)
    assert len(on_disk) == 1
    assert on_disk[0]["id"] == payload["id"]


def test_writer_uses_bureau_fallback(tmp_path: Path) -> None:
    sid = "S234"
    runs_root = tmp_path / "runs"

    summary_payload = {
        "validation_requirements": {
            "requirements": [
                {
                    "field": "account_status",
                    "category": "status",
                    "strength": "soft",
                    "ai_needed": True,
                    "notes": "status mismatch",
                }
            ],
            "field_consistency": {},
        }
    }
    bureaus_payload = {
        "transunion": {"account_status": "Open"},
    }

    account_dir = runs_root / sid / "cases" / "accounts" / "7"
    _write_json(account_dir / "summary.json", summary_payload)
    _write_json(account_dir / "bureaus.json", bureaus_payload)

    writer = ValidationPackWriter(sid, runs_root=runs_root)
    lines = writer.write_pack_for_account(7)

    assert len(lines) == 1
    payload = lines[0].payload
    assert payload["strength"] == "weak"
    assert payload["bureaus"]["transunion"]["raw"] == "Open"
    assert payload["context"]["requirement_note"] == "status mismatch"

    pack_path = runs_root / sid / "ai_packs" / "validation" / "packs" / "val_acc_007.jsonl"
    on_disk = _read_jsonl(pack_path)
    assert on_disk[0]["bureaus"]["transunion"]["raw"] == "Open"


def test_writer_skips_strong_fields(tmp_path: Path) -> None:
    sid = "S345"
    runs_root = tmp_path / "runs"

    summary_payload = {
        "validation_requirements": {
            "requirements": [
                {
                    "field": "payment_status",
                    "category": "status",
                    "strength": "strong",
                    "ai_needed": True,
                }
            ],
            "field_consistency": {},
        }
    }

    account_dir = runs_root / sid / "cases" / "accounts" / "9"
    _write_json(account_dir / "summary.json", summary_payload)

    writer = ValidationPackWriter(sid, runs_root=runs_root)
    lines = writer.write_pack_for_account(9)

    assert lines == []

    pack_path = runs_root / sid / "ai_packs" / "validation" / "packs" / "val_acc_009.jsonl"
    assert pack_path.exists()
    assert pack_path.read_text(encoding="utf-8") == ""


def test_writer_updates_index(tmp_path: Path) -> None:
    sid = "S456"
    runs_root = tmp_path / "runs"

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
                    "field": "account_status",
                    "category": "status",
                    "strength": "weak",
                    "ai_needed": True,
                },
            ],
            "field_consistency": {
                "balance_owed": {
                    "raw": {"transunion": "$100"},
                },
                "account_status": {
                    "raw": {"transunion": "Open"},
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
    assert entry["result_json"] == "results/acc_001.result.json"
    assert entry["result_jsonl"] == "results/acc_001.result.jsonl"
    assert entry["lines"] == 2
    assert entry["weak_fields"] == ["balance_owed", "account_status"]
    assert entry["status"] == "built"
    assert isinstance(entry["built_at"], str) and entry["built_at"].endswith("Z")
    assert isinstance(entry.get("source_hash"), str) and len(entry["source_hash"]) == 64

    # Update summary to remove one requirement and rebuild.
    summary_payload["validation_requirements"]["requirements"] = [
        summary_payload["validation_requirements"]["requirements"][0]
    ]
    _write_json(account_dir / "summary.json", summary_payload)

    writer.write_pack_for_account(1)

    refreshed_index = _read_index(index_path)
    assert len(refreshed_index["packs"]) == 1
    refreshed_entry = refreshed_index["packs"][0]
    assert refreshed_entry["lines"] == 1
    assert refreshed_entry["weak_fields"] == ["balance_owed"]
    assert isinstance(refreshed_entry.get("source_hash"), str)
    assert refreshed_entry["source_hash"] != entry["source_hash"]


def _seed_validation_account(
    runs_root: Path,
    sid: str,
    account_id: int,
    *,
    field: str = "balance_owed",
) -> None:
    account_dir = runs_root / sid / "cases" / "accounts" / str(account_id)
    summary_payload = {
        "validation_requirements": {
            "requirements": [
                {
                    "field": field,
                    "category": "activity",
                    "strength": "weak",
                    "ai_needed": True,
                }
            ],
            "field_consistency": {
                field: {"raw": {"transunion": "$100"}},
            },
        }
    }
    bureaus_payload = {
        "transunion": {field: "$100"},
        "experian": {field: "$105"},
    }
    _write_json(account_dir / "summary.json", summary_payload)
    _write_json(account_dir / "bureaus.json", bureaus_payload)


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
                "field": "balance_owed",
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

    stored_payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert stored_payload["sid"] == sid
    assert stored_payload["account_id"] == account_id
    assert stored_payload["status"] == "done"
    assert stored_payload["model"] == "gpt-infer"
    assert stored_payload["request_lines"] == len(lines)
    assert stored_payload["raw_response"] == {"id": "resp_123"}
    assert isinstance(stored_payload["completed_at"], str)
    assert stored_payload["results"] == [
        {
            "id": "acc_004__balance_owed",
            "account_id": account_id,
            "field": "balance_owed",
            "decision": "strong",
            "rationale": "values diverge",
            "citations": [],
        }
    ]

    results_dir = validation_results_dir(sid, runs_root=runs_root)
    jsonl_path = (
        results_dir / validation_result_jsonl_filename_for_account(account_id)
    )
    jsonl_lines = _read_jsonl(jsonl_path)
    assert jsonl_lines == stored_payload["results"]

    index_path = validation_index_path(sid, runs_root=runs_root)
    index_payload = _read_index(index_path)
    entry = _index_entry_for_account(index_payload, account_id)
    assert entry["status"] == "done"
    assert entry["request_lines"] == len(lines)
    assert entry["model"] == "gpt-infer"
    assert entry.get("error") is None
    assert isinstance(entry["completed_at"], str)
    assert isinstance(entry["sent_at"], str)
    assert isinstance(entry.get("source_hash"), str)


def test_store_validation_result_error(tmp_path: Path) -> None:
    sid = "S789"
    runs_root = tmp_path / "runs"
    account_id = 5

    _seed_validation_account(runs_root, sid, account_id, field="account_status")

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

    stored_payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert stored_payload["status"] == "error"
    assert stored_payload["error"] == "api_timeout"
    assert stored_payload["results"] == []

    index_path = validation_index_path(sid, runs_root=runs_root)
    entry = _index_entry_for_account(_read_index(index_path), account_id)
    assert entry["status"] == "error"
    assert entry["error"] == "api_timeout"
    assert isinstance(entry["completed_at"], str)
    assert isinstance(entry.get("source_hash"), str)


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
            "requirements": [
                {
                    "field": "balance_owed",
                    "category": "activity",
                    "strength": "weak",
                    "ai_needed": True,
                }
            ],
            "field_consistency": {
                "balance_owed": {"raw": {"transunion": "$100"}}
            },
        }
    }
    bureaus_payload = {"transunion": {"balance_owed": "$100"}}
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


def test_build_validation_packs_for_run_auto_send(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sid = "S601"
    runs_root = tmp_path / "runs"
    account_dir = runs_root / sid / "cases" / "accounts" / "1"
    account_dir.mkdir(parents=True, exist_ok=True)

    summary_payload = {
        "validation_requirements": {
            "requirements": [
                {
                    "field": "balance_owed",
                    "category": "activity",
                    "strength": "weak",
                    "ai_needed": True,
                }
            ],
            "field_consistency": {
                "balance_owed": {"raw": {"transunion": "$100"}}
            },
        }
    }
    bureaus_payload = {"transunion": {"balance_owed": "$100"}}

    _write_json(account_dir / "summary.json", summary_payload)
    _write_json(account_dir / "bureaus.json", bureaus_payload)

    monkeypatch.setenv("AUTO_VALIDATION_SEND", "1")
    monkeypatch.delenv("ENABLE_VALIDATION_SENDER", raising=False)
    monkeypatch.delenv("VALIDATION_SEND_ON_BUILD", raising=False)
    monkeypatch.setitem(
        sys.modules,
        "requests",
        types.SimpleNamespace(post=lambda *args, **kwargs: None),
    )

    captured: dict[str, Any] = {}

    def _fake_send(manifest: Any) -> list[dict[str, Any]]:
        captured["manifest"] = manifest
        return []

    monkeypatch.setattr(
        "backend.validation.send_packs.send_validation_packs",
        _fake_send,
    )

    build_validation_packs_for_run(sid, runs_root=runs_root)

    assert "manifest" in captured
    expected_index = validation_index_path(sid, runs_root=runs_root)
    assert Path(captured["manifest"]) == expected_index


def test_rewrite_index_to_canonical_layout(tmp_path: Path) -> None:
    sid = "S777"
    runs_root = tmp_path / "runs"

    account_dir = runs_root / sid / "cases" / "accounts" / "1"
    summary_payload = {
        "validation_requirements": {
            "requirements": [
                {
                    "field": "balance_owed",
                    "category": "activity",
                    "strength": "weak",
                    "ai_needed": True,
                }
            ],
            "field_consistency": {
                "balance_owed": {
                    "raw": {"transunion": "$100"},
                }
            },
        }
    }
    bureaus_payload = {
        "transunion": {"balance_owed": "$100"},
        "experian": {"balance_owed": "$105"},
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
    entry["result_jsonl"] = "../cases/accounts/1/result.jsonl"
    entry["result_json"] = "../cases/accounts/1/result.json"
    _write_json(index_path, payload)

    _, changed = rewrite_index_to_canonical_layout(index_path, runs_root=runs_root)
    assert changed is True

    rewritten = _read_index(index_path)
    assert rewritten["packs_dir"] == "packs"
    assert rewritten["results_dir"] == "results"

    rewritten_entry = _index_entry_for_account(rewritten, 1)
    assert rewritten_entry["pack"] == "packs/val_acc_001.jsonl"
    assert rewritten_entry["result_jsonl"] == "results/acc_001.result.jsonl"
    assert rewritten_entry["result_json"] == "results/acc_001.result.json"

    _, unchanged = rewrite_index_to_canonical_layout(index_path, runs_root=runs_root)
    assert unchanged is False
