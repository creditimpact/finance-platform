"""Minimal runflow smoke tests for merge, validation, and frontend stages."""

from __future__ import annotations

import importlib
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def runflow_env(tmp_path, monkeypatch) -> SimpleNamespace:
    """Return a helper namespace with reloaded runflow modules."""

    runs_root = tmp_path / "runs"
    monkeypatch.setenv("RUNS_ROOT", str(runs_root))
    monkeypatch.setenv("RUNFLOW_EVENTS", "1")
    monkeypatch.setenv("RUNFLOW_VERBOSE", "1")

    # Some pipeline modules expect ``requests`` to be available at import time.
    monkeypatch.setitem(sys.modules, "requests", types.ModuleType("requests"))

    runflow_steps = importlib.import_module("backend.core.runflow_steps")
    runflow = importlib.import_module("backend.core.runflow")
    runflow_io = importlib.import_module("backend.core.runflow.io")
    auto_ai = importlib.import_module("backend.pipeline.auto_ai")
    auto_ai_tasks = importlib.import_module("backend.pipeline.auto_ai_tasks")
    frontend_generator = importlib.import_module("backend.frontend.packs.generator")

    for module in (
        runflow_steps,
        runflow,
        runflow_io,
        auto_ai,
        auto_ai_tasks,
        frontend_generator,
    ):
        importlib.reload(module)

    return SimpleNamespace(
        runs_root=runs_root,
        runflow=runflow,
        runflow_steps=runflow_steps,
        runflow_io=runflow_io,
        auto_ai=auto_ai,
        auto_ai_tasks=auto_ai_tasks,
        frontend_generator=frontend_generator,
    )


def _load_steps(runs_root: Path, sid: str) -> dict[str, object]:
    steps_path = runs_root / sid / "runflow_steps.json"
    return json.loads(steps_path.read_text(encoding="utf-8"))


def _load_events(runs_root: Path, sid: str) -> list[dict[str, object]]:
    events_path = runs_root / sid / "runflow_events.jsonl"
    if not events_path.exists():
        return []
    return [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()]


def test_merge_empty_summary_has_no_packs(runflow_env: SimpleNamespace) -> None:
    sid = "SID-merge-empty"
    payload: dict[str, object] = {"sid": sid}
    tasks = runflow_env.auto_ai_tasks

    payload = tasks._merge_build_stage(payload)
    payload = tasks._merge_send_stage(payload)
    payload = tasks._merge_compact_stage(payload)
    tasks._finalize_stage(payload)

    steps_payload = _load_steps(runflow_env.runs_root, sid)
    merge_stage = steps_payload["stages"]["merge"]
    summary = merge_stage["summary"]
    assert summary["empty_ok"] is True

    step_names = [entry["name"] for entry in merge_stage.get("steps", [])]
    assert "pack_create" not in step_names
    assert "pack_skip" not in step_names

    events = _load_events(runflow_env.runs_root, sid)
    assert all(event.get("step") != "pack_skip" for event in events)


def test_merge_pack_create_without_skips(runflow_env: SimpleNamespace, monkeypatch: pytest.MonkeyPatch) -> None:
    from backend.core.ai.paths import ensure_merge_paths

    sid = "SID-merge-pack"
    runs_root = runflow_env.runs_root
    tasks = runflow_env.auto_ai_tasks

    monkeypatch.setattr(tasks, "has_ai_merge_best_pairs", lambda *_args, **_kwargs: True)

    def _fake_build_ai_packs(sid: str, runs_root: Path | str) -> None:
        root = Path(runs_root)
        paths = ensure_merge_paths(root, sid, create=True)
        pack_path = paths.packs_dir / "pair_000_001.jsonl"
        pack_path.write_text(json.dumps({"accounts": []}), encoding="utf-8")
        index_payload = {"pairs": [{"a": 0, "b": 1, "pack_path": str(pack_path)}]}
        paths.index_file.write_text(json.dumps(index_payload), encoding="utf-8")
        pairs_index = {"totals": {"scored_pairs": 1}}
        (paths.base / "pairs_index.json").write_text(json.dumps(pairs_index), encoding="utf-8")
        runflow_env.runflow.runflow_step(
            sid,
            "merge",
            "pack_create",
            account="0-1",
            out={"path": str(pack_path)},
        )

    monkeypatch.setattr(tasks, "_build_ai_packs", _fake_build_ai_packs)
    monkeypatch.setattr(tasks, "_send_ai_packs", lambda *_args, **_kwargs: None)

    payload: dict[str, object] = {"sid": sid}
    payload = tasks._merge_build_stage(payload)
    payload = tasks._merge_send_stage(payload)
    payload = tasks._merge_compact_stage(payload)
    tasks._finalize_stage(payload)

    steps_payload = _load_steps(runs_root, sid)
    merge_stage = steps_payload["stages"]["merge"]
    summary = merge_stage["summary"]
    assert summary["packs_created"] == 1
    assert summary["empty_ok"] is False

    step_names = [entry["name"] for entry in merge_stage.get("steps", [])]
    assert "pack_create" in step_names
    assert "pack_skip" not in step_names

    events = _load_events(runs_root, sid)
    assert all(event.get("step") != "pack_skip" for event in events)


def test_validation_summary_marks_empty_ok(runflow_env: SimpleNamespace) -> None:
    sid = "SID-validation-empty"

    runflow_env.auto_ai.run_validation_requirements_for_all_accounts(
        sid, runs_root=runflow_env.runs_root
    )

    steps_payload = _load_steps(runflow_env.runs_root, sid)
    validation_stage = steps_payload["stages"]["validation"]
    summary = validation_stage["summary"]
    assert summary["ai_packs_built"] == 0
    assert summary["empty_ok"] is True


def _write_summary(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_frontend_skips_accounts_missing_bureaus(runflow_env: SimpleNamespace) -> None:
    sid = "SID-frontend-skip"
    runs_root = runflow_env.runs_root
    accounts_root = runs_root / sid / "cases" / "accounts"

    valid_summary = {"account_id": "A-1", "labels": {"creditor": "Bank"}}
    valid_bureaus = {"transunion": {"account_number_display": "1111"}}
    _write_summary(accounts_root / "0" / "summary.json", valid_summary)
    _write_summary(accounts_root / "0" / "bureaus.json", valid_bureaus)

    skipped_summary = {"account_id": "B-1", "labels": {"creditor": "Store"}}
    _write_summary(accounts_root / "1" / "summary.json", skipped_summary)

    result = runflow_env.frontend_generator.generate_frontend_packs_for_run(
        sid, runs_root=runs_root
    )

    assert result["packs_count"] == 1

    index_path = runs_root / sid / "frontend" / "review" / "index.json"
    index_payload = json.loads(index_path.read_text(encoding="utf-8"))
    assert index_payload["packs_count"] == 1
    assert len(index_payload["packs"]) == 1

    steps_payload = _load_steps(runs_root, sid)
    frontend_stage = steps_payload["stages"]["frontend"]
    summary = frontend_stage["summary"]
    assert summary["packs_count"] == 1
    assert summary.get("skipped_missing") == 1


def test_frontend_error_records_runflow_stage_error(
    runflow_env: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    sid = "SID-frontend-error"
    runs_root = runflow_env.runs_root
    accounts_root = runs_root / sid / "cases" / "accounts"

    summary_payload = {"account_id": "ERR-1", "labels": {"creditor": "Error Bank"}}
    bureaus_payload = {"equifax": {"account_number_display": "9999"}}
    _write_summary(accounts_root / "0" / "summary.json", summary_payload)
    _write_summary(accounts_root / "0" / "bureaus.json", bureaus_payload)

    def _boom(*_args, **_kwargs) -> None:  # pragma: no cover - exercised via exception
        raise RuntimeError("boom")

    monkeypatch.setattr(runflow_env.frontend_generator, "_atomic_write_json", _boom)

    with pytest.raises(RuntimeError):
        runflow_env.frontend_generator.generate_frontend_packs_for_run(
            sid, runs_root=runs_root
        )

    steps_payload = _load_steps(runs_root, sid)
    frontend_stage = steps_payload["stages"]["frontend"]
    assert frontend_stage["status"] == "error"
    summary = frontend_stage["summary"]
    assert "error" in summary
    assert summary["error"]["type"] == "RuntimeError"
    finish_steps = [
        step
        for step in frontend_stage["steps"]
        if step.get("name") == "frontend_review_finish"
    ]
    assert finish_steps, "frontend_review_finish step should be recorded"
    assert any(step.get("status") == "error" for step in finish_steps)
    error_entry = next(
        step for step in finish_steps if step.get("status") == "error"
    )
    assert error_entry.get("out", {}).get("account_id") == "ERR-1"
    assert error_entry.get("out", {}).get("error_class") == "RuntimeError"

