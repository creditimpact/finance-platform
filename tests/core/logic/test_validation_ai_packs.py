from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.core.ai.paths import (
    ensure_validation_account_paths,
    ensure_validation_paths,
    validation_base_dir,
    validation_index_path,
    validation_logs_path,
    validation_pack_filename_for_account,
    validation_packs_dir,
    validation_result_filename_for_account,
    validation_results_dir,
)
from backend.core.logic import validation_ai_packs
from backend.core.logic.validation_ai_packs import (
    ValidationPacksConfig,
    build_validation_ai_packs_for_accounts,
    load_validation_packs_config,
)
from backend.pipeline.runs import RUNS_ROOT_ENV
from tests.helpers.fake_ai_client import FakeAIClient


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_validation_path_helpers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sid = "sid-paths"
    runs_root = tmp_path / "runs"
    monkeypatch.setenv(RUNS_ROOT_ENV, str(runs_root))

    base_dir = validation_base_dir(sid)
    assert base_dir == (runs_root / sid / "ai_packs" / "validation").resolve()

    packs_dir = validation_packs_dir(sid)
    results_dir = validation_results_dir(sid)
    assert packs_dir == (base_dir / "packs").resolve()
    assert results_dir == (base_dir / "results").resolve()
    assert packs_dir.exists() and results_dir.exists()

    index_path = validation_index_path(sid)
    logs_path = validation_logs_path(sid)
    assert index_path == (base_dir / "index.json").resolve()
    assert logs_path == (base_dir / "logs.txt").resolve()

    assert validation_pack_filename_for_account(7) == "val_acc_007.jsonl"
    assert (
        validation_pack_filename_for_account("15") == "val_acc_015.jsonl"
    )
    assert validation_result_filename_for_account(7) == "acc_007.result.json"
    assert validation_result_filename_for_account("15") == "acc_015.result.json"


def test_builder_creates_validation_structure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sid = "sid-validation"
    runs_root = tmp_path / "runs"
    monkeypatch.setenv(RUNS_ROOT_ENV, str(runs_root))

    build_validation_ai_packs_for_accounts(
        sid,
        account_indices=[14, "15", "14"],
        runs_root=runs_root,
    )

    validation_paths = ensure_validation_paths(runs_root, sid, create=False)
    base_dir = validation_paths.base

    created_indices = {14, 15}
    for idx in created_indices:
        account_paths = ensure_validation_account_paths(
            validation_paths, idx, create=False
        )
        assert account_paths.account_id == int(idx)
        assert account_paths.pack_file.exists()
        assert account_paths.prompt_file.exists()
        assert account_paths.result_summary_file.exists()

        pack_lines = [
            json.loads(line)
            for line in _read(account_paths.pack_file).splitlines()
            if line
        ]
        assert pack_lines == []
        model_results = json.loads(_read(account_paths.result_summary_file))
        assert model_results["status"] == "skipped"
        assert model_results["reason"] == "no_weak_items"
        assert model_results["model"] == "gpt-4o-mini"
        assert isinstance(model_results["timestamp"], str)
        assert model_results["duration_ms"] == 0
        assert _read(account_paths.prompt_file) == ""

    manifest_path = runs_root / sid / "manifest.json"
    assert manifest_path.exists()

    manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
    validation_section = manifest_data["ai"]["validation"]
    assert validation_section["base"] == str(base_dir.resolve())
    assert validation_section["dir"] == str(base_dir.resolve())
    assert validation_section["accounts"] == str(base_dir.resolve())
    assert validation_section["accounts_dir"] == str(base_dir.resolve())
    assert isinstance(validation_section["last_prepared_at"], str)

    packs_validation = manifest_data["ai"]["packs"]["validation"]
    assert packs_validation["base"] == str(base_dir.resolve())
    assert packs_validation["dir"] == str(base_dir.resolve())
    assert packs_validation["packs"] == str(validation_paths.packs_dir)
    assert packs_validation["packs_dir"] == str(validation_paths.packs_dir)
    assert packs_validation["results"] == str(validation_paths.results_dir)
    assert packs_validation["results_dir"] == str(validation_paths.results_dir)
    expected_index = validation_paths.index_file.resolve()
    assert packs_validation["index"] == str(expected_index)
    assert packs_validation["logs"] == str(validation_paths.log_file)
    assert isinstance(packs_validation["last_built_at"], str)

    index_payload = json.loads(expected_index.read_text(encoding="utf-8"))
    assert index_payload["sid"] == sid
    assert index_payload["schema_version"] == 1
    pack_accounts = {entry["account_id"] for entry in index_payload["packs"]}
    assert pack_accounts == {14, 15}
    for entry in index_payload["packs"]:
        assert entry["status"] == "skipped"
        assert entry["weak_fields"] == []
        assert entry["lines"] == 0
        assert "request_lines" not in entry
        assert isinstance(entry.get("source_hash"), str) and len(entry["source_hash"]) == 64
        assert isinstance(entry["built_at"], str)
        assert Path(entry["pack_path"]) == (
            validation_paths.packs_dir
            / validation_pack_filename_for_account(entry["account_id"])
        ).resolve()

    log_path = validation_paths.log_file
    assert log_path.exists()
    log_lines = [line for line in log_path.read_text(encoding="utf-8").splitlines() if line]
    assert len(log_lines) == 2
    parsed_logs = [json.loads(line) for line in log_lines]
    assert {entry["account_index"] for entry in parsed_logs} == {14, 15}
    for log_entry in parsed_logs:
        assert log_entry["weak_count"] == 0
        assert log_entry["statuses"] == ["pack_written", "no_weak_items"]
        assert log_entry["inference_status"] == "skipped"
        assert log_entry["inference_reason"] == "no_weak_items"
        assert log_entry["model"] == "gpt-4o-mini"


def test_load_validation_packs_config_defaults(tmp_path: Path) -> None:
    config = load_validation_packs_config(tmp_path / "missing")
    assert isinstance(config, ValidationPacksConfig)
    assert config.enable_write is True
    assert config.enable_infer is True
    assert config.model == "gpt-4o-mini"
    assert config.weak_limit == 0
    assert config.max_attempts >= 1
    assert config.backoff_seconds


def test_builder_populates_pack_and_preserves_prompt_and_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sid = "sid-existing"
    runs_root = tmp_path / "runs"
    monkeypatch.setenv(RUNS_ROOT_ENV, str(runs_root))

    validation_paths = ensure_validation_paths(runs_root, sid, create=True)
    account_paths = ensure_validation_account_paths(
        validation_paths, 42, create=True
    )

    account_paths.pack_file.write_text("", encoding="utf-8")
    account_paths.prompt_file.write_text("Existing prompt", encoding="utf-8")
    account_paths.result_summary_file.write_text(
        "{\"status\": \"done\"}\n", encoding="utf-8"
    )

    accounts_root = runs_root / sid / "cases" / "accounts"
    account_dir = accounts_root / "42"
    account_dir.mkdir(parents=True, exist_ok=True)
    summary_payload = {
        "validation_requirements": {
            "requirements": [
                {
                    "field": "balance_owed",
                    "category": "activity",
                    "min_days": 30,
                    "documents": ["statement"],
                    "ai_needed": True,
                },
                {
                    "field": "payment_status",
                    "category": "status",
                    "min_days": 10,
                    "documents": [],
                    "ai_needed": False,
                },
            ],
            "field_consistency": {
                "balance_owed": {
                    "consensus": "split",
                    "disagreeing_bureaus": ["experian"],
                    "missing_bureaus": ["equifax"],
                    "raw": {
                        "transunion": "100",
                        "experian": "150",
                        "equifax": None,
                    },
                    "normalized": {
                        "transunion": 100,
                        "experian": 150,
                        "equifax": None,
                    },
                }
            },
        }
    }
    (account_dir / "summary.json").write_text(
        json.dumps(summary_payload), encoding="utf-8"
    )

    fake_ai = FakeAIClient()
    fake_ai.add_response(
        json.dumps(
            {
                "sid": sid,
                "account_index": 42,
                "decisions": [],
            }
        )
    )

    build_validation_ai_packs_for_accounts(
        sid,
        account_indices=[42],
        runs_root=runs_root,
        ai_client=fake_ai,
    )

    pack_lines = [
        json.loads(line)
        for line in _read(account_paths.pack_file).splitlines()
        if line
    ]
    assert len(pack_lines) == 1
    pack_entry = pack_lines[0]
    assert pack_entry["field"] == "balance_owed"
    assert pack_entry["account_id"] == 42
    assert pack_entry["documents"] == ["statement"]
    assert pack_entry["consensus"] == "split"
    assert pack_entry["disagreeing_bureaus"] == ["experian"]
    assert pack_entry["missing_bureaus"] == ["equifax"]
    assert pack_entry["values"] == {
        "transunion": {"raw": "100", "normalized": 100},
        "experian": {"raw": "150", "normalized": 150},
        "equifax": {"raw": None, "normalized": None},
    }
    legacy_item = {
        key: value
        for key, value in pack_entry.items()
        if key not in {"account_id", "sid", "field_index"}
    }
    expected_prompt = validation_ai_packs._render_prompt(sid, 42, [legacy_item])
    assert _read(account_paths.prompt_file) == expected_prompt
    model_results = json.loads(_read(account_paths.result_summary_file))
    assert model_results["status"] == "ok"
    assert model_results["model"] == "gpt-4o-mini"

    index_payload = json.loads(
        (validation_paths.base / "index.json").read_text(encoding="utf-8")
    )
    packs_map = {entry["account_id"]: entry for entry in index_payload["packs"]}
    account_entry = packs_map[42]
    assert account_entry["status"] == model_results["status"]
    assert isinstance(account_entry["built_at"], str)
    assert Path(account_entry["pack_path"]).resolve() == account_paths.pack_file.resolve()
    assert account_entry["weak_fields"] == ["balance_owed"]
    assert account_entry["lines"] == 1
    assert isinstance(account_entry.get("source_hash"), str)
    assert len(account_entry["source_hash"]) == 64
    assert isinstance(model_results["timestamp"], str)
    assert isinstance(model_results["duration_ms"], int)
    assert model_results["response"] == {
        "sid": sid,
        "account_index": 42,
        "decisions": [],
    }

    log_lines = [
        line
        for line in validation_paths.log_file.read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert len(log_lines) == 1
    log_entry = json.loads(log_lines[0])
    assert log_entry["account_index"] == 42
    assert log_entry["weak_count"] == 1
    assert log_entry["statuses"] == ["pack_written", "infer_done"]
    assert log_entry["inference_status"] == "ok"
    assert "inference_reason" not in log_entry
    assert log_entry["model"] == "gpt-4o-mini"
    assert model_results["raw"] == json.dumps(
        {
            "sid": sid,
            "account_index": 42,
            "decisions": [],
        }
    )
    assert len(fake_ai.chat_payloads) == 1
    sent_payload = fake_ai.chat_payloads[0]
    assert sent_payload["model"] == "gpt-4o-mini"
    assert sent_payload["prompt"] == _read(account_paths.prompt_file)
    assert sent_payload["response_format"] == {"type": "json_object"}


def test_builder_skips_when_source_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sid = "sid-up-to-date"
    runs_root = tmp_path / "runs"
    monkeypatch.setenv(RUNS_ROOT_ENV, str(runs_root))

    accounts_root = runs_root / sid / "cases" / "accounts"
    account_dir = accounts_root / "11"
    account_dir.mkdir(parents=True, exist_ok=True)
    summary_payload = {
        "validation_requirements": {
            "requirements": [
                {
                    "field": "balance_owed",
                    "category": "activity",
                    "min_days": 30,
                    "documents": ["statement"],
                    "ai_needed": True,
                }
            ],
            "field_consistency": {
                "balance_owed": {
                    "consensus": "split",
                    "disagreeing_bureaus": ["experian"],
                    "missing_bureaus": [],
                    "raw": {"transunion": "100", "experian": "150"},
                    "normalized": {"transunion": 100, "experian": 150},
                }
            },
        }
    }
    (account_dir / "summary.json").write_text(
        json.dumps(summary_payload), encoding="utf-8"
    )

    validation_paths = ensure_validation_paths(runs_root, sid, create=True)
    account_paths = ensure_validation_account_paths(validation_paths, 11, create=True)

    fake_ai_first = FakeAIClient()
    fake_ai_first.add_response(
        json.dumps({"sid": sid, "account_index": 11, "decisions": []})
    )

    build_validation_ai_packs_for_accounts(
        sid,
        account_indices=[11],
        runs_root=runs_root,
        ai_client=fake_ai_first,
    )

    initial_pack_contents = account_paths.pack_file.read_text(encoding="utf-8")
    index_path = validation_paths.index_file
    initial_index = json.loads(index_path.read_text(encoding="utf-8"))
    initial_entry = {entry["account_id"]: entry for entry in initial_index["packs"]}[11]

    fake_ai_second = FakeAIClient()
    build_validation_ai_packs_for_accounts(
        sid,
        account_indices=[11],
        runs_root=runs_root,
        ai_client=fake_ai_second,
    )

    assert fake_ai_second.chat_payloads == []
    assert account_paths.pack_file.read_text(encoding="utf-8") == initial_pack_contents

    updated_index = json.loads(index_path.read_text(encoding="utf-8"))
    updated_entry = {entry["account_id"]: entry for entry in updated_index["packs"]}[11]
    assert updated_entry["status"] == initial_entry["status"] == "ok"
    assert updated_entry["built_at"] == initial_entry["built_at"]
    assert updated_entry["source_hash"] == initial_entry["source_hash"]

    log_lines = [
        json.loads(line)
        for line in validation_paths.log_file.read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert len(log_lines) == 2
    statuses = [entry["statuses"] for entry in log_lines]
    assert ["pack_written", "infer_done"] in statuses
    assert ["up_to_date"] in statuses


def test_builder_uses_configured_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sid = "sid-config"
    runs_root = tmp_path / "runs"
    monkeypatch.setenv(RUNS_ROOT_ENV, str(runs_root))

    validation_paths = ensure_validation_paths(runs_root, sid, create=True)
    (validation_paths.base / "ai_packs_config.yml").write_text(
        "model: gpt-validation-test\n", encoding="utf-8"
    )

    account_paths = ensure_validation_account_paths(
        validation_paths, 7, create=True
    )

    accounts_root = runs_root / sid / "cases" / "accounts"
    account_dir = accounts_root / "7"
    account_dir.mkdir(parents=True, exist_ok=True)
    summary_payload = {
        "validation_requirements": {
            "requirements": [
                {
                    "field": "account_status",
                    "category": "status",
                    "min_days": 10,
                    "documents": ["statement"],
                    "ai_needed": True,
                }
            ],
            "field_consistency": {},
        }
    }
    (account_dir / "summary.json").write_text(
        json.dumps(summary_payload), encoding="utf-8"
    )

    fake_ai = FakeAIClient()
    fake_ai.add_response(
        json.dumps(
            {
                "sid": sid,
                "account_index": 7,
                "decisions": [],
            }
        )
    )

    build_validation_ai_packs_for_accounts(
        sid,
        account_indices=[7],
        runs_root=runs_root,
        ai_client=fake_ai,
    )

    sent_payload = fake_ai.chat_payloads[0]
    assert sent_payload["model"] == "gpt-validation-test"

    results_payload = json.loads(_read(account_paths.result_summary_file))
    assert results_payload["model"] == "gpt-validation-test"


def test_builder_skips_when_write_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sid = "sid-disabled"
    runs_root = tmp_path / "runs"
    monkeypatch.setenv(RUNS_ROOT_ENV, str(runs_root))

    base_dir = runs_root / sid / "ai_packs" / "validation"
    base_dir.mkdir(parents=True, exist_ok=True)
    (base_dir / "ai_packs_config.yml").write_text(
        "validation_packs:\n  enable_write: false\n  enable_infer: true\n",
        encoding="utf-8",
    )

    accounts_root = runs_root / sid / "cases" / "accounts"
    account_dir = accounts_root / "5"
    account_dir.mkdir(parents=True, exist_ok=True)
    summary_payload = {
        "validation_requirements": {
            "requirements": [
                {
                    "field": "account_status",
                    "category": "status",
                    "min_days": 3,
                    "documents": ["statement"],
                    "ai_needed": True,
                }
            ],
            "field_consistency": {},
        }
    }
    (account_dir / "summary.json").write_text(
        json.dumps(summary_payload), encoding="utf-8"
    )

    fake_ai = FakeAIClient()

    build_validation_ai_packs_for_accounts(
        sid,
        account_indices=[5],
        runs_root=runs_root,
        ai_client=fake_ai,
    )

    assert not fake_ai.chat_payloads
    account_pack_dir = base_dir / "5"
    assert not account_pack_dir.exists()


def test_builder_skips_inference_when_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sid = "sid-no-infer"
    runs_root = tmp_path / "runs"
    monkeypatch.setenv(RUNS_ROOT_ENV, str(runs_root))

    validation_paths = ensure_validation_paths(runs_root, sid, create=True)
    (validation_paths.base / "ai_packs_config.yml").write_text(
        "validation_packs:\n  enable_infer: false\n",
        encoding="utf-8",
    )

    account_paths = ensure_validation_account_paths(
        validation_paths, 3, create=True
    )

    accounts_root = runs_root / sid / "cases" / "accounts"
    account_dir = accounts_root / "3"
    account_dir.mkdir(parents=True, exist_ok=True)
    summary_payload = {
        "validation_requirements": {
            "requirements": [
                {
                    "field": "account_status",
                    "category": "status",
                    "min_days": 3,
                    "documents": ["statement"],
                    "ai_needed": True,
                }
            ],
            "field_consistency": {},
        }
    }
    (account_dir / "summary.json").write_text(
        json.dumps(summary_payload), encoding="utf-8"
    )

    fake_ai = FakeAIClient()

    build_validation_ai_packs_for_accounts(
        sid,
        account_indices=[3],
        runs_root=runs_root,
        ai_client=fake_ai,
    )

    assert not fake_ai.chat_payloads
    results_payload = json.loads(_read(account_paths.result_summary_file))
    assert results_payload["status"] == "skipped"
    assert results_payload["reason"] == "inference_disabled"
    assert results_payload["attempts"] == 0


def test_builder_honors_weak_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sid = "sid-weak-limit"
    runs_root = tmp_path / "runs"
    monkeypatch.setenv(RUNS_ROOT_ENV, str(runs_root))

    validation_paths = ensure_validation_paths(runs_root, sid, create=True)
    (validation_paths.base / "ai_packs_config.yml").write_text(
        "validation_packs:\n  weak_limit: 1\n",
        encoding="utf-8",
    )

    account_paths = ensure_validation_account_paths(
        validation_paths, 9, create=True
    )

    accounts_root = runs_root / sid / "cases" / "accounts"
    account_dir = accounts_root / "9"
    account_dir.mkdir(parents=True, exist_ok=True)
    summary_payload = {
        "validation_requirements": {
            "requirements": [
                {
                    "field": "balance_owed",
                    "category": "activity",
                    "min_days": 30,
                    "documents": ["statement"],
                    "ai_needed": True,
                },
                {
                    "field": "account_status",
                    "category": "status",
                    "min_days": 10,
                    "documents": ["statement"],
                    "ai_needed": True,
                },
            ],
            "field_consistency": {},
        }
    }
    (account_dir / "summary.json").write_text(
        json.dumps(summary_payload), encoding="utf-8"
    )

    fake_ai = FakeAIClient()
    fake_ai.add_response(
        json.dumps(
            {
                "sid": sid,
                "account_index": 9,
                "decisions": [],
            }
        )
    )

    build_validation_ai_packs_for_accounts(
        sid,
        account_indices=[9],
        runs_root=runs_root,
        ai_client=fake_ai,
    )

    pack_lines = [
        json.loads(line)
        for line in _read(account_paths.pack_file).splitlines()
        if line
    ]
    assert len(pack_lines) == 1
    prompt_payload = _read(account_paths.prompt_file)
    assert "balance_owed" in prompt_payload or "account_status" in prompt_payload
    assert fake_ai.chat_payloads
