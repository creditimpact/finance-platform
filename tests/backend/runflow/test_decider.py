from __future__ import annotations

import importlib
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from typing import Mapping, Sequence


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

from backend import config
import backend.core.runflow as runflow_module
import backend.runflow.decider as runflow_decider
from backend.runflow.counters import note_style_stage_counts
from backend.core.ai.paths import ensure_note_style_paths


def _load_runflow(tmp_root: Path, sid: str) -> dict:
    path = tmp_root / sid / "runflow.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _write_note_style_index(
    run_dir: Path, entries: Sequence[Mapping[str, object]], totals: Mapping[str, int] | None = None
) -> None:
    base = run_dir / "ai_packs" / "note_style"
    base.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        "schema_version": 1,
        "packs": list(entries),
    }
    if totals is not None:
        payload["totals"] = dict(totals)
    (base / "index.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def test_decide_next_completes_when_no_accounts(tmp_path):
    runs_root = tmp_path / "runs"
    sid = "S-no-accounts"

    runflow_decider.record_stage(
        sid,
        "merge",
        status="success",
        counts={"count": 0},
        empty_ok=True,
        runs_root=runs_root,
    )

    decision = runflow_decider.decide_next(sid, runs_root=runs_root)
    assert decision["next"] == "complete_no_action"
    data = _load_runflow(runs_root, sid)
    assert data["run_state"] == "COMPLETE_NO_ACTION"


def test_decide_next_completes_when_validation_has_no_findings(tmp_path):
    runs_root = tmp_path / "runs"
    sid = "S-no-findings"

    runflow_decider.record_stage(
        sid,
        "merge",
        status="success",
        counts={"count": 2},
        empty_ok=False,
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

    decision = runflow_decider.decide_next(sid, runs_root=runs_root)
    assert decision["next"] == "complete_no_action"
    data = _load_runflow(runs_root, sid)
    assert data["run_state"] == "COMPLETE_NO_ACTION"


def test_decide_next_runs_frontend_then_moves_to_await_input(tmp_path):
    runs_root = tmp_path / "runs"
    sid = "S-with-findings"

    runflow_decider.record_stage(
        sid,
        "merge",
        status="success",
        counts={"count": 2},
        empty_ok=False,
        runs_root=runs_root,
    )
    runflow_decider.record_stage(
        sid,
        "validation",
        status="built",
        counts={"findings_count": 5},
        empty_ok=False,
        runs_root=runs_root,
    )

    decision = runflow_decider.decide_next(sid, runs_root=runs_root)
    assert decision["next"] == "gen_frontend_packs"
    data = _load_runflow(runs_root, sid)
    assert data["run_state"] == "VALIDATING"

    packs_dir = runs_root / sid / "frontend" / "review" / "packs"
    packs_dir.mkdir(parents=True, exist_ok=True)
    for index in range(2):
        (packs_dir / f"pack-{index}.json").write_text("{}", encoding="utf-8")

    runflow_decider.record_stage(
        sid,
        "frontend",
        status="published",
        counts={"packs_count": 2},
        empty_ok=False,
        runs_root=runs_root,
    )

    follow_up = runflow_decider.decide_next(sid, runs_root=runs_root)
    assert follow_up["next"] == "await_input"
    assert follow_up["reason"] == "frontend_published"
    data = _load_runflow(runs_root, sid)
    assert data["run_state"] == "AWAITING_CUSTOMER_INPUT"


def test_decide_next_frontend_zero_packs_marks_complete(tmp_path):
    runs_root = tmp_path / "runs"
    sid = "S-frontend-empty"

    runflow_decider.record_stage(
        sid,
        "merge",
        status="success",
        counts={"count": 2},
        empty_ok=False,
        runs_root=runs_root,
    )
    runflow_decider.record_stage(
        sid,
        "validation",
        status="built",
        counts={"findings_count": 3},
        empty_ok=False,
        runs_root=runs_root,
    )

    initial = runflow_decider.decide_next(sid, runs_root=runs_root)
    assert initial["next"] == "gen_frontend_packs"

    runflow_decider.record_stage(
        sid,
        "frontend",
        status="published",
        counts={"packs_count": 0},
        empty_ok=True,
        runs_root=runs_root,
    )

    follow_up = runflow_decider.decide_next(sid, runs_root=runs_root)
    assert follow_up == {
        "next": "complete_no_action",
        "reason": "frontend_no_packs",
    }
    data = _load_runflow(runs_root, sid)
    assert data["run_state"] == "COMPLETE_NO_ACTION"


def test_decide_next_stops_on_error(tmp_path):
    runs_root = tmp_path / "runs"
    sid = "S-error"

    runflow_decider.record_stage(
        sid,
        "validation",
        status="error",
        counts={"findings_count": 1},
        empty_ok=False,
        runs_root=runs_root,
    )

    decision = runflow_decider.decide_next(sid, runs_root=runs_root)
    assert decision["next"] == "stop_error"
    data = _load_runflow(runs_root, sid)
    assert data["run_state"] == "ERROR"


def test_decide_next_records_runflow_decide_step(tmp_path, monkeypatch):
    runs_root = tmp_path / "runs"
    sid = "S-runflow-decide"

    monkeypatch.setenv("RUNFLOW_VERBOSE", "1")
    monkeypatch.setenv("RUNFLOW_EVENTS", "1")
    monkeypatch.setenv("RUNS_ROOT", str(runs_root))

    importlib.reload(runflow_module)
    importlib.reload(runflow_decider)

    runflow_decider.record_stage(
        sid,
        "merge",
        status="success",
        counts={"count": 0},
        empty_ok=True,
        runs_root=runs_root,
    )

    decision = runflow_decider.decide_next(sid, runs_root=runs_root)
    assert decision == {"next": "complete_no_action", "reason": "no_accounts"}

    steps_path = runs_root / sid / "runflow_steps.json"
    payload = json.loads(steps_path.read_text(encoding="utf-8"))
    merge_stage = payload["stages"]["merge"]
    step_entries = [
        entry for entry in merge_stage.get("steps", []) if entry.get("name") == "runflow_decide"
    ]
    assert len(step_entries) == 1
    decision_entry = step_entries[0]
    assert decision_entry.get("out") == {"next": "done", "reason": "no_accounts"}

    # Reset modules to default configuration for other tests
    monkeypatch.setenv("RUNFLOW_VERBOSE", "0")
    monkeypatch.setenv("RUNFLOW_EVENTS", "0")
    monkeypatch.delenv("RUNS_ROOT", raising=False)
    importlib.reload(runflow_module)
    importlib.reload(runflow_decider)


def test_note_style_stage_promotion_requires_completed_results(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    sid = "SID-note-style"
    run_dir = runs_root / sid
    run_dir.mkdir(parents=True, exist_ok=True)

    initial_entries = [
        {"account_id": "idx-1", "status": "completed"},
        {"account_id": "idx-2", "status": "built"},
    ]
    _write_note_style_index(run_dir, initial_entries)

    data: dict[str, object] = {"sid": sid, "stages": {}}
    updated, promoted, log_context = runflow_decider._apply_note_style_stage_promotion(
        data, run_dir
    )

    assert updated is True
    assert promoted is False
    assert log_context == {"total": 2, "completed": 1, "failed": 0}

    stage_payload = data["stages"]["note_style"]
    assert stage_payload["status"] == "built"
    assert stage_payload["empty_ok"] is False
    assert stage_payload["metrics"] == {"packs_total": 2}
    assert stage_payload["results"] == {"results_total": 2, "completed": 1, "failed": 0}
    summary = stage_payload["summary"]
    assert summary["packs_total"] == 2
    assert summary["results_total"] == 2
    assert summary["completed"] == 1
    assert summary["failed"] == 0
    assert summary["empty_ok"] is False

    completed_entries = [
        {"account_id": "idx-1", "status": "completed"},
        {"account_id": "idx-2", "status": "completed"},
    ]
    _write_note_style_index(run_dir, completed_entries)

    updated_again, promoted_again, log_context_again = (
        runflow_decider._apply_note_style_stage_promotion(data, run_dir)
    )

    assert updated_again is True
    assert promoted_again is True
    assert log_context_again == {"total": 2, "completed": 2, "failed": 0}

    final_stage = data["stages"]["note_style"]
    assert final_stage["status"] == "success"
    assert final_stage["empty_ok"] is False
    assert final_stage["results"]["completed"] == 2
    assert final_stage["results"]["failed"] == 0
    assert final_stage["summary"]["completed"] == 2
    assert final_stage["summary"]["empty_ok"] is False


def test_note_style_stage_promotion_empty_index_marks_success(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    sid = "SID-note-empty"
    run_dir = runs_root / sid
    run_dir.mkdir(parents=True, exist_ok=True)

    _write_note_style_index(run_dir, [], totals={"total": 0, "completed": 0, "failed": 0})

    data: dict[str, object] = {"sid": sid, "stages": {}}
    updated, promoted, log_context = runflow_decider._apply_note_style_stage_promotion(
        data, run_dir
    )

    assert updated is True
    assert promoted is True
    assert log_context == {"total": 0, "completed": 0, "failed": 0}

    stage_payload = data["stages"]["note_style"]
    assert stage_payload["status"] == "success"
    assert stage_payload["empty_ok"] is True
    assert stage_payload["metrics"] == {"packs_total": 0}
    assert stage_payload["results"] == {"results_total": 0, "completed": 0, "failed": 0}
    assert stage_payload["summary"]["empty_ok"] is True
    assert stage_payload["summary"]["results_total"] == 0
    assert stage_payload["summary"]["completed"] == 0


def test_note_style_stage_promotion_error_on_failed_result(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    sid = "SID-note-failure"
    run_dir = runs_root / sid
    run_dir.mkdir(parents=True, exist_ok=True)

    entries = [
        {"account_id": "idx-1", "status": "completed"},
        {"account_id": "idx-2", "status": "failed"},
    ]
    _write_note_style_index(run_dir, entries)

    data: dict[str, object] = {"sid": sid, "stages": {}}
    updated, promoted, log_context = runflow_decider._apply_note_style_stage_promotion(
        data, run_dir
    )

    assert updated is True
    assert promoted is False
    assert log_context == {"total": 2, "completed": 1, "failed": 1}

    stage_payload = data["stages"]["note_style"]
    assert stage_payload["status"] == "error"
    assert stage_payload["empty_ok"] is False
    assert stage_payload["metrics"] == {"packs_total": 2}
    assert stage_payload["results"] == {"results_total": 2, "completed": 1, "failed": 1}
    assert stage_payload["summary"]["failed"] == 1


def test_note_style_stage_promotion_uses_manifest_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NOTE_STYLE_USE_MANIFEST_PATHS", "1")
    monkeypatch.setattr(config, "NOTE_STYLE_USE_MANIFEST_PATHS", True)

    runs_root = tmp_path / "runs"
    sid = "SID-note-manifest"
    run_dir = runs_root / sid
    run_dir.mkdir(parents=True, exist_ok=True)

    custom_base = run_dir / "style-data"
    custom_results = custom_base / "custom-results"
    custom_base.mkdir(parents=True, exist_ok=True)
    custom_results.mkdir(parents=True, exist_ok=True)

    manifest_payload = {
        "ai": {
            "packs": {
                "note_style": {
                    "base": "style-data",
                    "results_dir": "style-data/custom-results",
                    "index": "style-data/index.json",
                }
            }
        }
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    index_payload = {
        "packs": [{"account_id": "idx-1", "status": "completed"}],
        "totals": {"total": 1, "completed": 1, "failed": 0},
    }
    (custom_base / "index.json").write_text(
        json.dumps(index_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    (custom_results / "acc_idx-1.result.jsonl").write_text(
        json.dumps({"status": "completed"}, ensure_ascii=False), encoding="utf-8"
    )

    paths = ensure_note_style_paths(run_dir.parent, run_dir.name, create=False)
    assert paths.results_dir == custom_results.resolve()
    assert runflow_decider._note_style_index_status_mapping(run_dir) == {
        "idx-1": "completed"
    }
    assert runflow_decider._note_style_counts_from_results_dir(run_dir) == (
        1,
        1,
        0,
    )

    data: dict[str, object] = {"sid": sid, "stages": {}}
    updated, promoted, log_context = runflow_decider._apply_note_style_stage_promotion(
        data, run_dir
    )

    assert updated is True
    assert promoted is True
    assert log_context == {"total": 1, "completed": 1, "failed": 0}

    stage_payload = data["stages"]["note_style"]
    assert stage_payload["status"] == "success"
    assert stage_payload["metrics"] == {"packs_total": 1}
    assert stage_payload["results"] == {"results_total": 1, "completed": 1, "failed": 0}


def test_note_style_stage_counts_uses_manifest_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NOTE_STYLE_USE_MANIFEST_PATHS", "1")
    monkeypatch.setattr(config, "NOTE_STYLE_USE_MANIFEST_PATHS", True)

    runs_root = tmp_path / "runs"
    sid = "SID-note-counts"
    run_dir = runs_root / sid
    run_dir.mkdir(parents=True, exist_ok=True)

    custom_base = run_dir / "style-alt"
    custom_results = custom_base / "results-alt"
    custom_packs = custom_base / "packs-alt"
    custom_base.mkdir(parents=True, exist_ok=True)
    custom_results.mkdir(parents=True, exist_ok=True)
    custom_packs.mkdir(parents=True, exist_ok=True)

    manifest_payload = {
        "ai": {
            "packs": {
                "note_style": {
                    "base": "style-alt",
                    "results_dir": "style-alt/results-alt",
                    "packs_dir": "style-alt/packs-alt",
                    "index": "style-alt/index.json",
                }
            }
        }
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    index_payload = {
        "packs": [{"account_id": "idx-9", "status": "completed"}],
        "totals": {"total": 1, "completed": 1, "failed": 0},
    }
    (custom_base / "index.json").write_text(
        json.dumps(index_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    (custom_results / "acc_idx-9.result.jsonl").write_text(
        json.dumps({"analysis": {"tone": "neutral"}}, ensure_ascii=False),
        encoding="utf-8",
    )

    paths = ensure_note_style_paths(run_dir.parent, run_dir.name, create=False)
    assert paths.results_dir == custom_results.resolve()

    counts = note_style_stage_counts(run_dir)
    assert counts == {"packs_total": 1, "packs_completed": 1, "packs_failed": 0}
