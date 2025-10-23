"""Regression tests for stage readiness and umbrella promotion flows."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


def _ensure_requests_stub() -> None:
    if "requests" in sys.modules:
        return

    module = ModuleType("requests")

    class _DummySession:
        def get(self, *_args, **_kwargs):
            return SimpleNamespace(status_code=200, headers={}, text="")

        def close(self) -> None:
            pass

    module.Session = _DummySession
    module.RequestException = Exception
    sys.modules["requests"] = module


_ensure_requests_stub()

from backend.frontend.packs.config import load_frontend_stage_config
from backend.runflow import decider
from backend.validation.index_schema import ValidationIndex, ValidationPackRecord


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_validation_index(
    run_dir: Path,
    sid: str,
    statuses: list[str],
    *,
    include_results: bool,
) -> None:
    validation_dir = run_dir / "ai_packs" / "validation"
    results_dir = validation_dir / "results"
    packs_dir = validation_dir / "packs"
    packs_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    for existing in results_dir.glob("*.result.json*"):
        if existing.is_file():
            existing.unlink()

    records: list[ValidationPackRecord] = []
    for index, status in enumerate(statuses, start=1):
        filename = f"idx-{index:03d}.result.jsonl"
        if include_results:
            results_dir.joinpath(filename).write_text("{}", encoding="utf-8")
        record = ValidationPackRecord(
            account_id=index,
            pack=f"packs/idx-{index:03d}.json",
            result_jsonl=None,
            result_json=f"results/{filename}",
            lines=1,
            status=status,
            built_at="2024-01-01T00:00:00Z",
        )
        records.append(record)

    index = ValidationIndex(
        index_path=validation_dir / "index.json",
        sid=sid,
        packs_dir="packs",
        results_dir="results",
        packs=records,
    )
    index.write()


def _write_frontend_pack_files(run_dir: Path, count: int) -> None:
    config = load_frontend_stage_config(run_dir)
    for idx in range(count):
        (config.packs_dir / f"pack-{idx:03d}.json").write_text("{}", encoding="utf-8")


def _write_frontend_response(run_dir: Path, account_id: str, *, status: str = "completed") -> None:
    config = load_frontend_stage_config(run_dir)
    responses_dir = config.responses_dir
    responses_dir.mkdir(parents=True, exist_ok=True)

    filename = decider._response_filename_for_account(account_id)  # type: ignore[attr-defined]
    payload = {
        "account_id": account_id,
        "status": status,
        "received_at": "2024-01-01T00:00:00Z",
        "answers": {"explanation": "ready"},
    }
    _write_json(responses_dir / filename, payload)


def _load_runflow(run_dir: Path) -> dict:
    return json.loads((run_dir / "runflow.json").read_text(encoding="utf-8"))


def test_validation_refresh_promotes_after_results(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runs_root = tmp_path / "runs"
    sid = "SID-validation-progress"
    run_dir = runs_root / sid

    monkeypatch.setattr(decider, "runflow_refresh_umbrella_barriers", lambda _sid: None)

    _write_validation_index(run_dir, sid, ["built"] * 15, include_results=False)
    decider.refresh_validation_stage_from_index(sid, runs_root=runs_root)

    runflow_path = run_dir / "runflow.json"
    assert not runflow_path.exists()

    _write_validation_index(run_dir, sid, ["completed"] * 15, include_results=True)
    decider.refresh_validation_stage_from_index(sid, runs_root=runs_root)

    payload = _load_runflow(run_dir)
    validation_stage = payload["stages"]["validation"]

    assert validation_stage["status"] == "success"
    summary = validation_stage["summary"]
    assert summary["results_total"] == 15
    assert summary["completed"] == 15
    assert summary["failed"] == 0
    assert summary["empty_ok"] is False


def test_validation_refresh_promotes_when_zero_packs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runs_root = tmp_path / "runs"
    sid = "SID-validation-empty"
    run_dir = runs_root / sid

    monkeypatch.setattr(decider, "runflow_refresh_umbrella_barriers", lambda _sid: None)

    _write_validation_index(run_dir, sid, [], include_results=True)
    decider.refresh_validation_stage_from_index(sid, runs_root=runs_root)

    payload = _load_runflow(run_dir)
    validation_stage = payload["stages"]["validation"]

    assert validation_stage["status"] == "success"
    assert validation_stage["empty_ok"] is True
    summary = validation_stage["summary"]
    assert summary["results_total"] == 0
    assert summary["completed"] == 0
    assert summary["empty_ok"] is True


def test_frontend_refresh_requires_all_answers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runs_root = tmp_path / "runs"
    sid = "SID-frontend-progress"
    run_dir = runs_root / sid

    monkeypatch.setattr(decider, "runflow_refresh_umbrella_barriers", lambda _sid: None)

    _write_frontend_pack_files(run_dir, 3)
    decider.refresh_frontend_stage_from_responses(sid, runs_root=runs_root)

    runflow_path = run_dir / "runflow.json"
    assert not runflow_path.exists()

    for index in range(1, 4):
        _write_frontend_response(run_dir, f"idx-{index:03d}")

    decider.refresh_frontend_stage_from_responses(sid, runs_root=runs_root)

    payload = _load_runflow(run_dir)
    frontend_stage = payload["stages"]["frontend"]

    assert frontend_stage["status"] == "success"
    metrics = frontend_stage["metrics"]
    assert metrics["answers_required"] == 3
    assert metrics["answers_received"] == 3
    summary = frontend_stage["summary"]
    assert summary["answers_required"] == 3
    assert summary["answers_received"] == 3
    assert summary["empty_ok"] is False


def test_frontend_refresh_promotes_when_zero_required(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runs_root = tmp_path / "runs"
    sid = "SID-frontend-empty"
    run_dir = runs_root / sid

    monkeypatch.setattr(decider, "runflow_refresh_umbrella_barriers", lambda _sid: None)

    # Ensure directories exist without writing packs or responses.
    load_frontend_stage_config(run_dir)
    decider.refresh_frontend_stage_from_responses(sid, runs_root=runs_root)

    payload = _load_runflow(run_dir)
    frontend_stage = payload["stages"]["frontend"]

    assert frontend_stage["status"] == "success"
    assert frontend_stage["empty_ok"] is True
    metrics = frontend_stage["metrics"]
    assert metrics["answers_required"] == 0
    assert metrics["answers_received"] == 0
    summary = frontend_stage["summary"]
    assert summary["answers_required"] == 0
    assert summary["answers_received"] == 0
    assert summary["empty_ok"] is True


def test_umbrella_all_ready_requires_merge(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runs_root = tmp_path / "runs"
    sid = "SID-umbrella-merge-required"
    run_dir = runs_root / sid

    monkeypatch.setattr(decider, "runflow_refresh_umbrella_barriers", lambda _sid: None)
    monkeypatch.setenv("UMBRELLA_REQUIRE_MERGE", "1")
    monkeypatch.setenv("MERGE_REQUIRED", "1")

    _write_validation_index(run_dir, sid, ["completed"] * 2, include_results=True)
    decider.refresh_validation_stage_from_index(sid, runs_root=runs_root)

    _write_frontend_pack_files(run_dir, 2)
    for index in range(1, 3):
        _write_frontend_response(run_dir, f"idx-{index:03d}")
    decider.refresh_frontend_stage_from_responses(sid, runs_root=runs_root)

    merge_dir = run_dir / "ai_packs" / "merge"
    _write_json(
        merge_dir / "pairs_index.json",
        {
            "schema_version": 2,
            "sid": sid,
            "root": ".",
            "totals": {"created_packs": 2},
            "pairs": [],
        },
    )

    statuses = decider.reconcile_umbrella_barriers(sid, runs_root=runs_root)
    assert statuses["merge_ready"] is False
    assert statuses["validation_ready"] is True
    assert statuses["review_ready"] is True
    assert statuses["style_ready"] is True
    assert statuses["all_ready"] is False

    results_dir = merge_dir / "results"
    packs_dir = merge_dir / "packs"
    results_dir.mkdir(parents=True, exist_ok=True)
    packs_dir.mkdir(parents=True, exist_ok=True)
    _write_json(results_dir / "pair-000.result.json", {"status": "completed"})
    packs_dir.joinpath("pair_000.jsonl").write_text("[]", encoding="utf-8")
    _write_json(
        merge_dir / "pairs_index.json",
        {
            "schema_version": 2,
            "sid": sid,
            "root": ".",
            "totals": {"created_packs": 1},
            "pairs": [
                {
                    "account_id": 1,
                    "pack": "packs/pair_000.jsonl",
                    "result_json": "results/pair-000.result.json",
                    "status": "completed",
                    "lines": 1,
                }
            ],
        },
    )

    statuses = decider.reconcile_umbrella_barriers(sid, runs_root=runs_root)
    assert statuses["merge_ready"] is True
    assert statuses["validation_ready"] is True
    assert statuses["review_ready"] is True
    assert statuses["style_ready"] is True
    assert statuses["all_ready"] is True

    payload = _load_runflow(run_dir)
    umbrella = payload["umbrella_barriers"]
    assert umbrella["merge_ready"] is True
    assert umbrella["style_ready"] is True
    assert umbrella["all_ready"] is True


def test_umbrella_all_ready_without_merge_requirement(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runs_root = tmp_path / "runs"
    sid = "SID-umbrella-merge-optional"
    run_dir = runs_root / sid

    monkeypatch.setattr(decider, "runflow_refresh_umbrella_barriers", lambda _sid: None)
    monkeypatch.setenv("UMBRELLA_REQUIRE_MERGE", "0")
    monkeypatch.setenv("MERGE_REQUIRED", "0")

    _write_validation_index(run_dir, sid, ["completed"] * 1, include_results=True)
    decider.refresh_validation_stage_from_index(sid, runs_root=runs_root)

    _write_frontend_pack_files(run_dir, 1)
    _write_frontend_response(run_dir, "idx-001")
    decider.refresh_frontend_stage_from_responses(sid, runs_root=runs_root)

    statuses = decider.reconcile_umbrella_barriers(sid, runs_root=runs_root)
    assert statuses["merge_ready"] is True
    assert statuses["validation_ready"] is True
    assert statuses["review_ready"] is True
    assert statuses["style_ready"] is True
    assert statuses["all_ready"] is True

    payload = _load_runflow(run_dir)
    assert payload["umbrella_ready"] is True
    umbrella = payload["umbrella_barriers"]
    assert umbrella["merge_ready"] is True
    assert umbrella["style_ready"] is True
    assert umbrella["all_ready"] is True

