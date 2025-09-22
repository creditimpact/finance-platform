import json
import logging
from pathlib import Path

import backend.api.tasks as task_module
from backend.api.tasks import extract_problematic_accounts
from backend.core.logic.report_analysis import problem_case_builder, problem_extractor
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
