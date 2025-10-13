from __future__ import annotations

import json
from pathlib import Path

import backend.frontend.packs.generator as generator_module
from backend.frontend.packs.generator import generate_frontend_packs_for_run


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_generate_frontend_packs_builds_account_pack(tmp_path):
    runs_root = tmp_path / "runs"
    sid = "S100"
    account_dir = runs_root / sid / "cases" / "accounts" / "1"

    summary_payload = {
        "account_id": "acct-1",
        "labels": {
            "creditor": "Sample Creditor",
            "account_type": {"normalized": "Credit Card"},
            "status": {"normalized": "Closed"},
        },
    }
    bureaus_payload = {
        "transunion": {
            "account_number_display": "****1234",
            "balance_owed": "$100",
            "date_opened": "2023-01-01",
            "closed_date": "2023-02-01",
            "date_reported": "2023-03-01",
            "account_status": "Closed",
            "account_type": "Credit Card",
        },
        "experian": {
            "account_number_display": "XXXX1234",
            "balance_owed": "$100",
            "date_opened": "2023-01-02",
            "closed_date": "--",
            "date_reported": "2023-03-02",
            "account_status": "Closed",
            "account_type": "Credit Card",
        },
    }
    meta_payload = {"heading_guess": "John Doe"}
    raw_lines_payload = [
        {"text": "JOHN DOE"},
        {"text": "Account # 1234"},
    ]
    tags_payload = [
        {"kind": "issue", "type": "wrong_account"},
        {"kind": "note", "type": "internal"},
        {"kind": "issue", "type": "late_payment"},
    ]

    _write_json(account_dir / "summary.json", summary_payload)
    _write_json(account_dir / "bureaus.json", bureaus_payload)
    _write_json(account_dir / "meta.json", meta_payload)
    _write_json(account_dir / "raw_lines.json", raw_lines_payload)
    _write_json(account_dir / "tags.json", tags_payload)

    result = generate_frontend_packs_for_run(sid, runs_root=runs_root)

    pack_path = runs_root / sid / "frontend" / "accounts" / "acct-1" / "pack.json"
    assert pack_path.exists()

    pack_payload = json.loads(pack_path.read_text(encoding="utf-8"))
    assert pack_payload["creditor_name"] == "Sample Creditor"
    assert pack_payload["account_type"] == "Credit Card"
    assert pack_payload["status"] == "Closed"
    assert pack_payload["last4"]["last4"] == "1234"
    assert pack_payload["balance_owed"]["consensus"] == "$100"
    assert set(pack_payload["balance_owed"]["per_bureau"].keys()) == {"transunion", "experian"}
    assert pack_payload["questions"][0]["id"] == "ownership"
    assert len(pack_payload["bureau_badges"]) == 2
    assert pack_payload["holder_name"] == "John Doe"
    assert pack_payload["primary_issue"] == "wrong_account"
    assert pack_payload["issues"] == ["wrong_account", "late_payment"]
    assert pack_payload["pointers"] == {
        "meta": "cases/accounts/1/meta.json",
        "tags": "cases/accounts/1/tags.json",
        "raw": "cases/accounts/1/raw_lines.json",
        "bureaus": "cases/accounts/1/bureaus.json",
        "flat": "cases/accounts/1/fields_flat.json",
        "summary": "cases/accounts/1/summary.json",
    }

    index_path = runs_root / sid / "frontend" / "index.json"
    assert index_path.exists()
    index_payload = json.loads(index_path.read_text(encoding="utf-8"))
    assert index_payload["packs_count"] == 1
    assert index_payload["accounts"][0]["pack_path"] == "frontend/accounts/acct-1/pack.json"
    assert index_payload["questions"][1]["id"] == "recognize"

    responses_dir = runs_root / sid / "frontend" / "responses"
    assert responses_dir.is_dir()
    assert not any(responses_dir.iterdir())

    assert result["status"] == "success"
    assert result["packs_count"] == 1
    assert result["empty_ok"] is False
    assert result["built"] is True
    assert result["packs_dir"] == str((runs_root / sid / "frontend").absolute())
    assert isinstance(result["last_built_at"], str)


def test_generate_frontend_packs_handles_missing_accounts(tmp_path):
    runs_root = tmp_path / "runs"
    sid = "S-empty"

    result = generate_frontend_packs_for_run(sid, runs_root=runs_root)

    index_path = runs_root / sid / "frontend" / "index.json"
    assert index_path.exists()
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    assert payload["accounts"] == []
    assert result["status"] == "success"
    assert result["packs_count"] == 0
    assert result["empty_ok"] is True
    assert result["built"] is True
    assert result["packs_dir"] == str((runs_root / sid / "frontend").absolute())
    assert isinstance(result["last_built_at"], str)


def test_generate_frontend_packs_respects_feature_flag(tmp_path, monkeypatch):
    runs_root = tmp_path / "runs"
    sid = "S-disabled"

    monkeypatch.setenv("ENABLE_FRONTEND_PACKS", "0")

    result = generate_frontend_packs_for_run(sid, runs_root=runs_root)

    assert result == {
        "status": "skipped",
        "packs_count": 0,
        "empty_ok": True,
        "built": False,
        "packs_dir": str((runs_root / sid / "frontend").absolute()),
        "last_built_at": None,
    }
    assert not (runs_root / sid).exists()


def test_generate_frontend_packs_task_exposed(tmp_path, monkeypatch):
    runs_root = tmp_path / "runs"
    sid = "S-task"

    monkeypatch.setenv("RUNS_ROOT", str(runs_root))

    import importlib
    import sys
    import types

    fake_requests = types.ModuleType("requests")
    monkeypatch.setitem(sys.modules, "requests", fake_requests)

    module_name = "backend.api.tasks"
    if module_name in sys.modules:
        api_tasks = importlib.reload(sys.modules[module_name])
    else:
        api_tasks = importlib.import_module(module_name)

    result = api_tasks.generate_frontend_packs_task.run(sid)

    assert result["packs_count"] == 0
    assert result["empty_ok"] is True

    index_path = runs_root / sid / "frontend" / "index.json"
    assert index_path.exists()


def test_generate_frontend_packs_continues_on_pack_write_failure(tmp_path, monkeypatch):
    runs_root = tmp_path / "runs"
    sid = "S-partial"

    account1_dir = runs_root / sid / "cases" / "accounts" / "1"
    account2_dir = runs_root / sid / "cases" / "accounts" / "2"

    shared_summary = {
        "account_id": "acct-success",
        "labels": {"creditor": "Cred", "account_type": "Loan", "status": "Open"},
    }
    shared_bureaus = {
        "transunion": {
            "account_number_display": "****9999",
            "balance_owed": "$50",
            "date_opened": "2023-01-01",
            "date_reported": "2023-02-01",
            "account_status": "Open",
            "account_type": "Loan",
        }
    }

    _write_json(account1_dir / "summary.json", {**shared_summary, "account_id": "acct-fail"})
    _write_json(account1_dir / "bureaus.json", shared_bureaus)
    _write_json(account1_dir / "tags.json", [{"kind": "issue", "type": "late_payment"}])

    _write_json(account2_dir / "summary.json", shared_summary)
    _write_json(account2_dir / "bureaus.json", shared_bureaus)
    _write_json(account2_dir / "tags.json", [{"kind": "issue", "type": "late_payment"}])

    original_write = generator_module._atomic_write_json

    def flaky_atomic_write(path, payload):
        if "acct-fail" in str(path):
            raise OSError("disk full")
        return original_write(path, payload)

    monkeypatch.setattr(generator_module, "_atomic_write_json", flaky_atomic_write)

    result = generator_module.generate_frontend_packs_for_run(sid, runs_root=runs_root)

    failing_pack = runs_root / sid / "frontend" / "accounts" / "acct-fail" / "pack.json"
    successful_pack = (
        runs_root / sid / "frontend" / "accounts" / "acct-success" / "pack.json"
    )

    assert not failing_pack.exists()
    assert successful_pack.exists()

    index_path = runs_root / sid / "frontend" / "index.json"
    index_payload = json.loads(index_path.read_text(encoding="utf-8"))

    assert index_payload["packs_count"] == 1
    assert index_payload["accounts"][0]["account_id"] == "acct-success"

    assert result["status"] == "success"
    assert result["packs_count"] == 1
