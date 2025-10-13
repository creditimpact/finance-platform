import json
import logging
import sys
import types
from pathlib import Path

sys.modules.setdefault("requests", types.ModuleType("requests"))

import backend.api.tasks as task_module
from backend.api.tasks import extract_problematic_accounts
from backend.core.logic.report_analysis import problem_case_builder, problem_extractor
from backend.runflow import manifest as runflow_manifest
from backend import settings


def _write_accounts(base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    data = {"accounts": [{"account_index": 1, "fields": {"past_due_amount": 50}}]}
    base.write_text(json.dumps(data), encoding="utf-8")


def test_extract_problematic_accounts_task_builder(tmp_path, monkeypatch, caplog):
    sid = "S777"

    # Redirect project root for all modules
    monkeypatch.setattr(settings, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(problem_case_builder, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(problem_extractor, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(task_module, "PROJECT_ROOT", tmp_path, raising=False)
    monkeypatch.setenv("ENABLE_AUTO_AI_PIPELINE", "1")

    task_module._AUTO_AI_PIPELINE_ENQUEUED.clear()

    ai_calls: list[str] = []

    class _DummyTask:
        def delay(self, sid_value: str):
            ai_calls.append(sid_value)
            return {"sid": sid_value}

    monkeypatch.setattr(task_module, "maybe_run_ai_pipeline_task", _DummyTask())
    monkeypatch.setattr(
        task_module, "has_ai_merge_best_tags", lambda sid: True
    )

    # Create Stage-A account artifacts
    acc_path = (
        tmp_path
        / "traces"
        / "blocks"
        / sid
        / "accounts_table"
        / "accounts_from_full.json"
    )
    _write_accounts(acc_path)

    def _run_build(prev, sid):
        summary = task_module.build_problem_cases_task(prev=prev, sid=sid)
        if isinstance(prev, dict):
            prev["summary"] = summary
        return summary

    monkeypatch.setattr(task_module.build_problem_cases_task, "delay", _run_build)

    caplog.set_level(logging.INFO)
    result = extract_problematic_accounts.run(sid)

    assert result["sid"] == sid
    assert len(result["found"]) == 1
    cand_id = result["found"][0]["account_id"]
    assert (
        tmp_path / "runs" / sid / "cases" / "accounts" / f"{cand_id}.json"
    ).exists()
    assert (tmp_path / "runs" / sid / "cases" / "index.json").exists()
    assert result["summary"]["problematic"] == 1
    assert any(f"PROBLEMATIC start sid={sid}" in m for m in caplog.messages)
    assert any(f"PROBLEMATIC done sid={sid} found=1" in m for m in caplog.messages)
    assert ai_calls == [sid]


def test_extract_problematic_accounts_task_no_candidates(tmp_path, monkeypatch, caplog):
    sid = "S000"

    monkeypatch.setattr(settings, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(problem_case_builder, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(problem_extractor, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(task_module, "PROJECT_ROOT", tmp_path, raising=False)
    monkeypatch.setenv("ENABLE_AUTO_AI_PIPELINE", "1")

    task_module._AUTO_AI_PIPELINE_ENQUEUED.clear()

    ai_calls: list[str] = []

    class _DummyTask:
        def delay(self, sid_value: str):
            ai_calls.append(sid_value)
            return {"sid": sid_value}

    monkeypatch.setattr(task_module, "maybe_run_ai_pipeline_task", _DummyTask())
    monkeypatch.setattr(
        task_module, "has_ai_merge_best_tags", lambda sid: False
    )

    acc_path = (
        tmp_path / "traces" / "blocks" / sid / "accounts_table" / "accounts_from_full.json"
    )
    acc_path.parent.mkdir(parents=True, exist_ok=True)
    acc_path.write_text(json.dumps({"accounts": [{"account_index": 1, "fields": {}}]}))

    def _run_build(prev, sid):
        summary = task_module.build_problem_cases_task(prev=prev, sid=sid)
        if isinstance(prev, dict):
            prev["summary"] = summary
        return summary

    monkeypatch.setattr(task_module.build_problem_cases_task, "delay", _run_build)

    caplog.set_level(logging.INFO)
    result = extract_problematic_accounts.run(sid)

    assert result["sid"] == sid
    assert result["found"] == []
    assert result["summary"]["problematic"] == 0
    index = tmp_path / "runs" / sid / "cases" / "index.json"
    assert index.exists()
    accounts_dir = tmp_path / "runs" / sid / "cases" / "accounts"
    assert accounts_dir.exists()
    assert sorted(p.name for p in accounts_dir.iterdir()) == ["index.json"]
    assert any(f"PROBLEMATIC start sid={sid}" in m for m in caplog.messages)
    assert any(f"PROBLEMATIC done sid={sid} found=0" in m for m in caplog.messages)
    assert ai_calls == []


def test_build_problem_cases_runs_frontend_even_when_not_requested(tmp_path, monkeypatch):
    sid = "S-frontend"
    runs_root = tmp_path / "runs"

    monkeypatch.setenv("RUNS_ROOT", str(runs_root))
    monkeypatch.setenv("ENABLE_AUTO_AI_PIPELINE", "0")
    monkeypatch.setattr(task_module, "ENABLE_VALIDATION_REQUIREMENTS", True)
    task_module._AUTO_AI_PIPELINE_ENQUEUED.clear()

    monkeypatch.setattr(settings, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(problem_case_builder, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(problem_extractor, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(task_module, "PROJECT_ROOT", tmp_path, raising=False)

    accounts_dir = runs_root / sid / "cases" / "accounts"
    accounts_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(task_module, "detect_problem_accounts", lambda _: [])

    def _fake_build_cases(sid_value: str, candidates: list[object]):
        (accounts_dir / "index.json").write_text("{}", encoding="utf-8")
        return {"cases": {"count": len(candidates), "dir": str(accounts_dir.parent)}}

    monkeypatch.setattr(task_module, "build_problem_cases", _fake_build_cases)
    monkeypatch.setattr(task_module, "detect_and_persist_date_convention", lambda *_: {})
    monkeypatch.setattr(
        task_module,
        "run_validation_requirements_for_all_accounts",
        lambda *_, **__: {"ok": True, "processed_accounts": 0, "findings_count": 0},
    )

    stage_calls: list[tuple[str, str, dict]] = []

    def _record_stage(sid_value: str, stage: str, **kwargs):
        stage_calls.append((sid_value, stage, kwargs))
        return {}

    monkeypatch.setattr(task_module, "record_stage", _record_stage)
    monkeypatch.setattr(
        task_module, "decide_next", lambda *_, **__: {"next": "stop_error"}
    )

    frontend_calls: list[tuple[str, object, bool]] = []

    def _fake_generate_frontend_packs(
        sid_value: str,
        *,
        runs_root: object | None = None,
        force: bool = False,
    ) -> dict:
        frontend_calls.append((sid_value, runs_root, force))
        packs_base = Path(runs_root) if runs_root else runs_root
        packs_dir = packs_base / sid_value / "frontend" if isinstance(packs_base, Path) else runs_root
        return {
            "status": "success",
            "packs_count": 0,
            "empty_ok": True,
            "built": True,
            "packs_dir": str(packs_dir) if packs_dir else None,
            "last_built_at": "now",
        }

    monkeypatch.setattr(task_module, "generate_frontend_packs_for_run", _fake_generate_frontend_packs)

    manifest_calls: list[tuple[tuple, dict]] = []

    def _record_manifest_update(*args, **kwargs):
        manifest_calls.append((args, kwargs))
        return runflow_manifest.update_manifest_frontend(*args, **kwargs)

    monkeypatch.setattr(task_module, "update_manifest_frontend", _record_manifest_update)

    task_module.build_problem_cases_task(prev={"sid": sid, "found": []}, sid=sid)

    assert frontend_calls, "frontend packs generator was not invoked"
    frontend_stage_entries = [entry for entry in stage_calls if entry[1] == "frontend"]
    assert frontend_stage_entries, "frontend stage was not recorded"
    assert frontend_stage_entries[0][2]["empty_ok"] is True
    assert manifest_calls, "frontend manifest update was not attempted"
