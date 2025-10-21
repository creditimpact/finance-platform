from __future__ import annotations

import importlib
import json
import sys
import types
from pathlib import Path

import backend.core.runflow as runflow_module
import backend.runflow.decider as runflow_decider


def _reload_modules(*modules):
    for module in modules:
        importlib.reload(module)


def _load_steps(runs_root: Path, sid: str) -> dict:
    return json.loads((runs_root / sid / "runflow_steps.json").read_text(encoding="utf-8"))


def _write_bureaus(runs_root: Path, sid: str, idx: int, payload: dict[str, object]) -> None:
    account_dir = runs_root / sid / "cases" / "accounts" / str(idx)
    account_dir.mkdir(parents=True, exist_ok=True)
    (account_dir / "bureaus.json").write_text(json.dumps(payload), encoding="utf-8")


def test_merge_stage_zero_pairs_marks_empty_ok(tmp_path, monkeypatch):
    runs_root = tmp_path / "runs"
    sid = "SID-merge-empty"

    monkeypatch.setenv("RUNS_ROOT", str(runs_root))
    monkeypatch.setenv("RUNFLOW_VERBOSE", "1")
    _reload_modules(runflow_module, runflow_decider)

    runflow_module.runflow_start_stage(sid, "merge")
    runflow_module.runflow_end_stage(
        sid,
        "merge",
        status="success",
        summary={"scored_pairs": 0, "empty_ok": True},
        stage_status="empty",
        empty_ok=True,
    )

    steps_payload = _load_steps(runs_root, sid)
    merge_stage = steps_payload["stages"]["merge"]
    assert merge_stage["status"] == "empty"
    assert merge_stage["empty_ok"] is True
    assert merge_stage["steps"] == []
    assert merge_stage["substages"]["default"]["steps"] == []


def test_validation_zero_findings_still_runs_frontend(tmp_path, monkeypatch):
    runs_root = tmp_path / "runs"
    sid = "SID-frontend-empty"

    monkeypatch.setenv("RUNS_ROOT", str(runs_root))
    monkeypatch.setenv("RUNFLOW_VERBOSE", "1")
    monkeypatch.setenv("RUNFLOW_EVENTS", "1")
    fake_requests = types.ModuleType("requests")
    monkeypatch.setitem(sys.modules, "requests", fake_requests)
    import backend.frontend.packs.generator as frontend_generator

    _reload_modules(runflow_module, runflow_decider, frontend_generator)

    runflow_decider.record_stage(
        sid,
        "merge",
        status="success",
        counts={"count": 0},
        empty_ok=True,
        runs_root=runs_root,
    )
    runflow_decider.record_stage(
        sid,
        "validation",
        status="built",
        counts={"findings_count": 0},
        empty_ok=True,
        runs_root=runs_root,
    )

    result = frontend_generator.generate_frontend_packs_for_run(sid, runs_root=runs_root)
    assert result["packs_count"] == 0
    assert result["empty_ok"] is True

    steps_payload = _load_steps(runs_root, sid)
    frontend_stage = steps_payload["stages"]["frontend"]
    assert frontend_stage["status"] == "empty"
    assert frontend_stage["empty_ok"] is True
    assert frontend_stage["summary"]["packs_count"] == 0

    step_names = [entry["name"] for entry in frontend_stage["steps"]]
    assert step_names == [
        "frontend_review_start",
        "frontend_review_no_candidates",
        "frontend_review_finish",
    ]

    validation_stage = steps_payload["stages"]["validation"]
    assert validation_stage["empty_ok"] is True
    assert validation_stage["summary"]["findings_count"] == 0


def test_merge_topn_steps_match_disk_counts(tmp_path, monkeypatch):
    runs_root = tmp_path / "runs"
    sid = "SID-merge-topn"

    monkeypatch.setenv("RUNS_ROOT", str(runs_root))
    monkeypatch.setenv("RUNFLOW_VERBOSE", "1")
    monkeypatch.setenv("RUNFLOW_EVENTS", "1")
    monkeypatch.setenv("RUNFLOW_STEPS_PAIR_TOPN", "2")
    fake_requests = types.ModuleType("requests")
    monkeypatch.setitem(sys.modules, "requests", fake_requests)
    import backend.core.logic.report_analysis.account_merge as account_merge

    _reload_modules(runflow_module, account_merge)

    _write_bureaus(
        runs_root,
        sid,
        0,
        {
            "transunion": {"account_number_display": "****1234"},
        },
    )
    _write_bureaus(
        runs_root,
        sid,
        1,
        {
            "transunion": {"account_number_display": "****5678"},
        },
    )
    _write_bureaus(
        runs_root,
        sid,
        2,
        {
            "transunion": {"account_number_display": "****9012"},
        },
    )

    account_merge.score_all_pairs_0_100(sid, [], runs_root=runs_root)

    steps_payload = _load_steps(runs_root, sid)
    merge_stage = steps_payload["stages"]["merge"]

    match_entries = [
        entry
        for entry in merge_stage["steps"]
        if entry.get("name") == "acctnum_match_level"
    ]
    assert len(match_entries) == 2
    assert [entry["metrics"]["rank"] for entry in match_entries] == [1, 2]

    summary_entry = next(
        entry
        for entry in merge_stage["steps"]
        if entry.get("name") == "acctnum_pairs_summary"
    )
    index_payload = json.loads(
        (runs_root / sid / "ai_packs" / "merge" / "pairs_index.json").read_text(encoding="utf-8")
    )
    assert summary_entry["metrics"]["scored_pairs"] == index_payload["totals"]["scored_pairs"]
    assert summary_entry["metrics"]["topn_limit"] == 2

