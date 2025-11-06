from __future__ import annotations

import importlib
import json
import os
from contextlib import contextmanager
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from typing import Iterator, Mapping, Sequence

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

    if "backend.ai" not in sys.modules:
        ai_path = Path(__file__).resolve().parents[3] / "backend" / "ai"
        ai_module = ModuleType("backend.ai")
        ai_module.__file__ = str(ai_path / "__init__.py")
        ai_module.__path__ = [str(ai_path)]
        ai_module.__package__ = "backend.ai"
        import importlib.machinery as _machinery

        spec = _machinery.ModuleSpec("backend.ai", loader=None, is_package=True)
        spec.submodule_search_locations = [str(ai_path)]
        ai_module.__spec__ = spec
        sys.modules["backend.ai"] = ai_module
    module.RequestException = Exception
    sys.modules["requests"] = module


_ensure_requests_stub()

from backend import config
import backend.core.runflow as runflow_module
import backend.runflow.decider as runflow_decider
from backend.runflow.counters import note_style_stage_counts
from backend.core.ai.paths import (
    ensure_note_style_paths,
    note_style_pack_filename,
    note_style_result_filename,
)


def _load_runflow(tmp_root: Path, sid: str) -> dict:
    path = tmp_root / sid / "runflow.json"
    return json.loads(path.read_text(encoding="utf-8"))


@contextmanager
def _zero_pack_finalization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    sid: str,
    skip_count: int = 2,
    emit_step: bool = True,
    skip_reason: str = "missing_original_creditor",
    merge_reason: str | None = None,
    include_merge_zero: bool = True,
) -> Iterator[SimpleNamespace]:
    global runflow_module, runflow_decider

    runs_root = tmp_path / "runs"
    monkeypatch.setenv("RUNS_ROOT", str(runs_root))
    monkeypatch.setenv("MERGE_ZERO_PACKS_SIGNAL", "1")
    monkeypatch.setenv("MERGE_SKIP_COUNTS_ENABLED", "1")
    monkeypatch.setenv("UMBRELLA_INCLUDE_MERGE_ZERO_PACKS_FLAG", "1")
    monkeypatch.setenv("RUNFLOW_VERBOSE", "1")
    monkeypatch.setenv("RUNFLOW_EVENTS", "1")
    monkeypatch.setenv("RUNFLOW_EMIT_ZERO_PACKS_STEP", "1" if emit_step else "0")

    runflow_module = importlib.reload(runflow_module)
    runflow_decider = importlib.reload(runflow_decider)

    merge_dir = runs_root / sid / "ai_packs" / "merge"
    results_dir = merge_dir / "results"
    packs_dir = merge_dir / "packs"
    results_dir.mkdir(parents=True, exist_ok=True)
    packs_dir.mkdir(parents=True, exist_ok=True)

    totals_payload: dict[str, object] = {
        "scored_pairs": skip_count,
        "created_packs": 0,
        "skip_counts": {skip_reason: skip_count},
        "skip_reason_top": skip_reason,
    }
    if include_merge_zero:
        totals_payload["merge_zero_packs"] = True
    if merge_reason is not None:
        totals_payload["reason"] = merge_reason

    index_payload = {
        "totals": totals_payload,
        "pairs": [],
    }
    index_path = merge_dir / "pairs_index.json"
    index_path.write_text(json.dumps(index_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    try:
        result = runflow_decider.finalize_merge_stage(sid, runs_root=runs_root)
        payload = _load_runflow(runs_root, sid)
        steps_path = runs_root / sid / "runflow_steps.json"
        steps_payload = (
            json.loads(steps_path.read_text(encoding="utf-8"))
            if steps_path.exists()
            else {}
        )
        yield SimpleNamespace(
            runs_root=runs_root,
            sid=sid,
            result=result,
            payload=payload,
            steps=steps_payload,
        )
    finally:
        runflow_module = importlib.reload(runflow_module)
        runflow_decider = importlib.reload(runflow_decider)


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


def _ensure_note_style_dirs(run_dir: Path) -> tuple[Path, Path]:
    paths = ensure_note_style_paths(run_dir.parent, run_dir.name, create=True)
    return paths.packs_dir, paths.results_dir


def _write_note_style_pack(run_dir: Path, account_id: str) -> Path:
    packs_dir, _ = _ensure_note_style_dirs(run_dir)
    pack_path = packs_dir / note_style_pack_filename(account_id)
    pack_path.parent.mkdir(parents=True, exist_ok=True)
    pack_path.write_text("{}\n", encoding="utf-8")
    return pack_path


def _write_note_style_result(
    run_dir: Path,
    account_id: str,
    *,
    status: str | None = None,
    error: Mapping[str, object] | None = None,
) -> Path:
    _, results_dir = _ensure_note_style_dirs(run_dir)
    result_path = results_dir / note_style_result_filename(account_id)
    payload: dict[str, object] = {}
    if status:
        payload["status"] = status
    if error:
        payload["error"] = dict(error)
    result_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return result_path


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


def test_finalize_persists_zero_packs_metrics(tmp_path, monkeypatch):
    with _zero_pack_finalization(
        tmp_path,
        monkeypatch,
        sid="S-zero-pack-metrics",
        skip_count=3,
        merge_reason="All pairs gated: missing original creditor",
    ) as ctx:
        metrics = ctx.result["metrics"]
        assert metrics["merge_zero_packs"] is True
        assert metrics["pairs_scored"] == 3
        assert metrics["created_packs"] == 0
        assert metrics["skip_reason_top"] == "missing_original_creditor"
        assert metrics["skip_counts"] == {"missing_original_creditor": 3}
        assert metrics["merge_reason"] == "All pairs gated: missing original creditor"

        merge_stage = ctx.payload["stages"]["merge"]
        stage_metrics = merge_stage["metrics"]
        assert stage_metrics["merge_zero_packs"] is True
        assert stage_metrics["skip_reason_top"] == "missing_original_creditor"
        assert stage_metrics["skip_counts"]["missing_original_creditor"] == 3
        assert stage_metrics["merge_reason"] == "All pairs gated: missing original creditor"

        summary = merge_stage["summary"]
        assert summary["merge_zero_packs"] is True
        assert summary["skip_reason_top"] == "missing_original_creditor"
        assert summary["skip_counts"] == {"missing_original_creditor": 3}
        assert summary["pairs_scored"] == 3
        assert summary["packs_created"] == 0
        assert summary["merge_reason"] == "All pairs gated: missing original creditor"
        summary_metrics = summary.get("metrics", {})
        assert summary_metrics.get("pairs_scored") == 3
        assert summary_metrics.get("created_packs") == 0
        assert summary_metrics.get("merge_zero_packs") is True

        barriers = ctx.payload["umbrella_barriers"]
        assert barriers["merge_zero_packs"] is True


def test_umbrella_barriers_include_merge_zero_packs_diagnostic(tmp_path, monkeypatch):
    with _zero_pack_finalization(
        tmp_path,
        monkeypatch,
        sid="S-zero-pack-barrier",
    ) as ctx:
        barriers = ctx.payload["umbrella_barriers"]
        assert barriers["merge_zero_packs"] is True
        assert barriers["merge_ready"] is True


def test_runflow_emits_merge_zero_packs_step(tmp_path, monkeypatch):
    with _zero_pack_finalization(
        tmp_path,
        monkeypatch,
        sid="S-zero-pack-step",
        emit_step=True,
    ) as ctx:
        merge_stage = ctx.steps.get("stages", {}).get("merge", {})
        zero_pack_entries = [
            entry for entry in merge_stage.get("steps", []) if entry.get("name") == "merge_zero_packs"
        ]
        assert len(zero_pack_entries) == 1
        assert zero_pack_entries[0].get("status") == "success"
        assert zero_pack_entries[0].get("out") == {
            "skip_reason_top": "missing_original_creditor",
            "skip_counts": {"missing_original_creditor": 2},
        }


def test_finalize_backfills_missing_merge_zero_packs(tmp_path, monkeypatch):
    with _zero_pack_finalization(
        tmp_path,
        monkeypatch,
        sid="S-zero-pack-backfill",
        include_merge_zero=False,
        skip_count=4,
    ) as ctx:
        metrics = ctx.payload["stages"]["merge"]["metrics"]
        summary = ctx.payload["stages"]["merge"]["summary"]
        assert metrics["merge_zero_packs"] is True
        assert summary["merge_zero_packs"] is True
        assert ctx.result["metrics"]["merge_zero_packs"] is True


def test_reconcile_backfills_zero_pack_metadata_from_index(tmp_path, monkeypatch):
    global runflow_module, runflow_decider

    runs_root = tmp_path / "runs"
    sid = "S-zero-reconcile"

    monkeypatch.setenv("RUNS_ROOT", str(runs_root))
    monkeypatch.setenv("UMBRELLA_INCLUDE_MERGE_ZERO_PACKS_FLAG", "1")
    monkeypatch.setenv("UMBRELLA_BARRIERS_ENABLED", "1")
    monkeypatch.setenv("UMBRELLA_BARRIERS_LOG", "0")
    monkeypatch.setenv("MERGE_ZERO_PACKS_SIGNAL", "1")
    monkeypatch.setenv("MERGE_SKIP_COUNTS_ENABLED", "1")

    umbrella_module = ModuleType("backend.runflow.umbrella")
    umbrella_module.schedule_merge_autosend = lambda *args, **kwargs: None
    umbrella_module.schedule_note_style_after_validation = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "backend.runflow.umbrella", umbrella_module)

    runflow_module = importlib.reload(runflow_module)
    runflow_decider = importlib.reload(runflow_decider)

    try:
        runflow_decider.record_stage(
            sid,
            "merge",
            status="success",
            counts={"pairs_scored": 0, "packs_created": 0},
            empty_ok=True,
            metrics={},
            runs_root=runs_root,
            refresh_barriers=False,
        )

        run_dir = runs_root / sid

        manifest_index = os.path.join("ai_packs", "merge", "pairs_index.json")
        manifest_payload = {"ai": {"packs": {"index": manifest_index}}}
        (run_dir / "manifest.json").write_text(
            json.dumps(manifest_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        index_path = run_dir / "ai_packs" / "merge" / "pairs_index.json"
        index_path.parent.mkdir(parents=True, exist_ok=True)
        index_payload = {
            "totals": {
                "merge_zero_packs": True,
                "scored_pairs": 7,
                "created_packs": 0,
                "skip_counts": {"missing_original_creditor": 7},
                "skip_reason_top": "missing_original_creditor",
                "reason": "All pairs gated: missing original creditor",
            },
            "pairs": [],
        }
        index_path.write_text(
            json.dumps(index_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        statuses = runflow_decider.reconcile_umbrella_barriers(sid, runs_root=runs_root)

        assert statuses["merge_zero_packs"] is True

        payload = _load_runflow(runs_root, sid)
        merge_stage = payload["stages"]["merge"]
        metrics = merge_stage["metrics"]
        summary = merge_stage["summary"]
        assert merge_stage["merge_zero_packs"] is True
        assert metrics["merge_zero_packs"] is True
        assert summary["merge_zero_packs"] is True
        assert metrics["pairs_scored"] == 7
        assert metrics["created_packs"] == 0
        assert metrics["skip_counts"] == {"missing_original_creditor": 7}
        assert metrics["skip_reason_top"] == "missing_original_creditor"
        assert summary.get("skip_counts") == {"missing_original_creditor": 7}
        assert summary.get("skip_reason_top") == "missing_original_creditor"
        assert metrics.get("reason") == "All pairs gated: missing original creditor"
        assert summary.get("merge_reason") == "All pairs gated: missing original creditor"

        umbrella_barriers = payload["umbrella_barriers"]
        assert umbrella_barriers["merge_zero_packs"] is True
    finally:
        runflow_module = importlib.reload(runflow_module)
        runflow_decider = importlib.reload(runflow_decider)


def test_resolve_merge_index_path_prefers_manifest(tmp_path, monkeypatch):
    sid = "S-index-manifest"
    run_dir = tmp_path / sid
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest_index = os.path.join("ai_packs", "merge", "pairs_index.json")
    manifest_payload = {"ai": {"packs": {"index": manifest_index}}}

    monkeypatch.setenv("MERGE_INDEX_PATH", str(tmp_path / "ignored.json"))
    monkeypatch.setattr(config, "MERGE_INDEX_PATH", "fallback/pairs_index.json", raising=False)

    expected = (run_dir / "ai_packs" / "merge" / "pairs_index.json").resolve()

    result = runflow_decider._resolve_merge_index_path(sid, run_dir, manifest_payload)

    assert result == expected


def test_resolve_merge_index_path_uses_env_override(tmp_path, monkeypatch):
    sid = "S-index-env"
    run_dir = tmp_path / sid
    run_dir.mkdir(parents=True, exist_ok=True)

    env_relative = os.path.join("custom", "pairs_index.json")
    expected = (run_dir / env_relative).resolve()

    monkeypatch.setenv("MERGE_INDEX_PATH", env_relative)
    monkeypatch.setattr(config, "MERGE_INDEX_PATH", "fallback/pairs_index.json", raising=False)

    result = runflow_decider._resolve_merge_index_path(sid, run_dir, None)

    assert result == expected


def test_runflow_events_include_zero_pack_summary(tmp_path, monkeypatch):
    with _zero_pack_finalization(
        tmp_path,
        monkeypatch,
        sid="S-zero-pack-events",
    ) as ctx:
        events_path = ctx.runs_root / ctx.sid / "runflow_events.jsonl"
        assert events_path.exists()
        events = [
            json.loads(line)
            for line in events_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        merge_end_events = [
            event for event in events if event.get("stage") == "merge" and event.get("event") == "end"
        ]
        assert merge_end_events, events
        final_event = merge_end_events[-1]
        summary = final_event.get("summary", {})
        assert summary.get("merge_zero_packs") is True
        assert summary.get("skip_reason_top") == "missing_original_creditor"
        assert summary.get("skip_counts") == {"missing_original_creditor": 2}
        metrics_payload = summary.get("metrics", {})
        assert metrics_payload.get("merge_zero_packs") is True
        assert metrics_payload.get("skip_reason_top") == "missing_original_creditor"


def test_decider_skips_merge_send_wait_on_zero_packs(tmp_path, monkeypatch):
    global runflow_module, runflow_decider

    runs_root = tmp_path / "runs"
    sid = "S-zero-fastpath"

    monkeypatch.setenv("RUNS_ROOT", str(runs_root))
    monkeypatch.setenv("MERGE_ZERO_PACKS_SIGNAL", "1")
    monkeypatch.setenv("RUNFLOW_MERGE_ZERO_PACKS_FASTPATH", "1")

    runflow_module = importlib.reload(runflow_module)
    runflow_decider = importlib.reload(runflow_decider)

    (runs_root / sid).mkdir(parents=True, exist_ok=True)

    runflow_decider.record_stage(
        sid,
        "merge",
        status="success",
        counts={"pairs_scored": 5, "packs_created": 0},
        empty_ok=True,
        metrics={
            "merge_zero_packs": True,
            "skip_counts": {"missing_original_creditor": 3},
            "skip_reason_top": "missing_original_creditor",
        },
        runs_root=runs_root,
    )

    decision = runflow_decider.decide_next(sid, runs_root=runs_root)

    assert decision["next"] == "run_validation"
    assert decision["reason"] == "merge_zero_packs"
    assert decision.get("skip_merge_wait") is True
    assert decision.get("merge_zero_packs") is True
    assert decision.get("skip_reason_top") == "missing_original_creditor"
    assert decision.get("skip_counts") == {"missing_original_creditor": 3}

    payload = _load_runflow(runs_root, sid)
    assert payload["run_state"] == "VALIDATING"

    runflow_module = importlib.reload(runflow_module)
    runflow_decider = importlib.reload(runflow_decider)


def test_decider_fastpath_persists_without_run_state_change(tmp_path, monkeypatch):
    global runflow_module, runflow_decider

    runs_root = tmp_path / "runs"
    sid = "S-fastpath-persist"

    monkeypatch.setenv("RUNS_ROOT", str(runs_root))
    monkeypatch.setenv("RUNFLOW_MERGE_ZERO_PACKS_FASTPATH", "1")
    monkeypatch.setenv("VALIDATION_AUTOSEND", "1")
    monkeypatch.setenv("VALIDATION_ZERO_PACKS_FASTPATH", "1")
    monkeypatch.setenv("UMBRELLA_INCLUDE_MERGE_ZERO_PACKS_FLAG", "1")
    monkeypatch.setenv("RUNFLOW_VERBOSE", "1")
    monkeypatch.setenv("RUNFLOW_EVENTS", "1")

    runflow_module = importlib.reload(runflow_module)
    runflow_decider = importlib.reload(runflow_decider)

    def fake_enqueue(enqueue_sid: str, run_dir: Path, *, merge_zero_packs: bool, payload=None) -> bool:
        runflow_module.runflow_step(
            enqueue_sid,
            "validation",
            "fastpath_send",
            status="queued",
            out={"merge_zero_packs": bool(merge_zero_packs)},
        )
        lock_path = run_dir / ".locks" / "validation_fastpath.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path.write_text("locked", encoding="utf-8")
        return True

    monkeypatch.setattr(runflow_decider, "_enqueue_validation_fastpath", fake_enqueue)

    runflow_decider.record_stage(
        sid,
        "merge",
        status="success",
        counts={"pairs_scored": 2, "packs_created": 0},
        empty_ok=True,
        metrics={
            "merge_zero_packs": True,
            "skip_counts": {"missing_original_creditor": 2},
            "skip_reason_top": "missing_original_creditor",
        },
        runs_root=runs_root,
    )

    runflow_decider.record_stage(
        sid,
        "validation",
        status="built",
        counts={"findings_count": 3},
        empty_ok=False,
        metrics={"packs_total": 3},
        runs_root=runs_root,
    )

    initial_snapshot = _load_runflow(runs_root, sid)
    assert initial_snapshot["run_state"] == "VALIDATING"

    decision = runflow_decider.decide_next(sid, runs_root=runs_root)

    assert decision["next"] == "run_validation"
    assert decision["reason"] == "merge_zero_packs"
    assert decision.get("skip_merge_wait") is True

    payload = _load_runflow(runs_root, sid)
    assert payload["run_state"] == "VALIDATING"
    validation_stage = payload["stages"]["validation"]
    assert validation_stage["sent"] is True
    assert validation_stage["status"] == "in_progress"
    assert validation_stage["metrics"]["merge_zero_packs"] is True
    assert validation_stage["metrics"]["skip_counts"] == {"missing_original_creditor": 2}
    assert validation_stage["metrics"]["skip_reason_top"] == "missing_original_creditor"
    assert validation_stage["skip_counts"] == {"missing_original_creditor": 2}
    assert validation_stage["skip_reason_top"] == "missing_original_creditor"
    assert isinstance(validation_stage["last_at"], str)
    assert validation_stage["summary"]["merge_zero_packs"] is True
    assert validation_stage["summary"]["skip_counts"] == {"missing_original_creditor": 2}
    assert validation_stage["summary"]["skip_reason_top"] == "missing_original_creditor"
    summary_metrics = validation_stage["summary"]["metrics"]
    assert summary_metrics["merge_zero_packs"] is True
    assert summary_metrics["skip_counts"] == {"missing_original_creditor": 2}
    assert summary_metrics["skip_reason_top"] == "missing_original_creditor"

    events_path = runs_root / sid / "runflow_events.jsonl"
    events_text = events_path.read_text(encoding="utf-8")
    assert '"fastpath_send"' in events_text
    assert '"status": "queued"' in events_text

    runflow_module = importlib.reload(runflow_module)
    runflow_decider = importlib.reload(runflow_decider)


def test_decider_fastpath_creates_validation_stage_when_missing(tmp_path, monkeypatch):
    global runflow_module, runflow_decider

    runs_root = tmp_path / "runs"
    sid = "S-fastpath-create"

    monkeypatch.setenv("RUNS_ROOT", str(runs_root))
    monkeypatch.setenv("RUNFLOW_MERGE_ZERO_PACKS_FASTPATH", "1")
    monkeypatch.setenv("VALIDATION_AUTOSEND", "1")
    monkeypatch.setenv("VALIDATION_ZERO_PACKS_FASTPATH", "1")
    monkeypatch.setenv("UMBRELLA_INCLUDE_MERGE_ZERO_PACKS_FLAG", "1")
    monkeypatch.setenv("RUNFLOW_VERBOSE", "1")

    runflow_module = importlib.reload(runflow_module)
    runflow_decider = importlib.reload(runflow_decider)

    enqueue_calls: list[tuple[str, Path, bool, dict[str, object]]] = []

    def fake_enqueue(enqueue_sid: str, run_dir: Path, *, merge_zero_packs: bool, payload=None) -> bool:
        enqueue_calls.append((enqueue_sid, run_dir, merge_zero_packs, dict(payload or {})))
        lock_path = run_dir / ".locks" / "validation_fastpath.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path.write_text("locked", encoding="utf-8")
        return True

    monkeypatch.setattr(runflow_decider, "_enqueue_validation_fastpath", fake_enqueue)

    runflow_decider.record_stage(
        sid,
        "merge",
        status="success",
        counts={"pairs_scored": 2, "packs_created": 0},
        empty_ok=True,
        metrics={
            "merge_zero_packs": True,
            "skip_counts": {"missing_original_creditor": 2},
            "skip_reason_top": "missing_original_creditor",
        },
        runs_root=runs_root,
    )

    decision = runflow_decider.decide_next(sid, runs_root=runs_root)

    assert decision["next"] == "run_validation"
    assert decision["reason"] == "merge_zero_packs"
    assert decision["validation_fastpath"] is True

    payload = _load_runflow(runs_root, sid)
    validation_stage = payload["stages"]["validation"]
    assert validation_stage["sent"] is True
    assert validation_stage["status"] == "in_progress"
    assert validation_stage["metrics"]["merge_zero_packs"] is True
    assert validation_stage["metrics"]["skip_counts"] == {"missing_original_creditor": 2}
    assert validation_stage["metrics"]["skip_reason_top"] == "missing_original_creditor"
    assert validation_stage["summary"]["merge_zero_packs"] is True
    assert validation_stage["summary"]["skip_counts"] == {"missing_original_creditor": 2}
    assert validation_stage["summary"]["skip_reason_top"] == "missing_original_creditor"
    assert validation_stage["summary"]["metrics"]["merge_zero_packs"] is True
    assert validation_stage["summary"]["metrics"]["skip_counts"] == {"missing_original_creditor": 2}
    assert validation_stage["summary"]["metrics"]["skip_reason_top"] == "missing_original_creditor"
    assert isinstance(validation_stage["last_at"], str)

    assert len(enqueue_calls) == 1
    assert enqueue_calls[0][0] == sid
    assert enqueue_calls[0][2] is True

    runflow_module = importlib.reload(runflow_module)
    runflow_decider = importlib.reload(runflow_decider)


def test_note_style_stage_promotion_requires_completed_results(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    sid = "SID-note-style"
    run_dir = runs_root / sid
    run_dir.mkdir(parents=True, exist_ok=True)

    initial_entries = [
        {"account_id": "idx-1", "status": "built"},
        {"account_id": "idx-2", "status": "built"},
    ]
    _write_note_style_index(run_dir, initial_entries)
    _write_note_style_pack(run_dir, "idx-1")
    _write_note_style_pack(run_dir, "idx-2")

    data: dict[str, object] = {"sid": sid, "stages": {}}
    updated, promoted, log_context = runflow_decider._apply_note_style_stage_promotion(
        data, run_dir
    )

    assert updated is True
    assert promoted is False
    assert log_context == {"total": 2, "completed": 0, "failed": 0}

    stage_payload = data["stages"]["note_style"]
    assert stage_payload["status"] == "built"
    assert stage_payload["empty_ok"] is False
    assert stage_payload["metrics"] == {"packs_total": 2}
    assert stage_payload["results"] == {"results_total": 2, "completed": 0, "failed": 0}
    summary = stage_payload["summary"]
    assert summary["packs_total"] == 2
    assert summary["results_total"] == 2
    assert summary["completed"] == 0
    assert summary["failed"] == 0
    assert summary["empty_ok"] is False
    assert stage_payload["sent"] is False
    assert stage_payload["completed_at"] is None

    _write_note_style_result(run_dir, "idx-1", status="completed")
    progress_entries = [
        {"account_id": "idx-1", "status": "completed"},
        {"account_id": "idx-2", "status": "built"},
    ]
    _write_note_style_index(run_dir, progress_entries)

    updated_again, promoted_again, log_context_again = (
        runflow_decider._apply_note_style_stage_promotion(data, run_dir)
    )

    assert updated_again is True
    assert promoted_again is False
    assert log_context_again == {"total": 2, "completed": 1, "failed": 0}

    mid_stage = data["stages"]["note_style"]
    assert mid_stage["status"] == "processing"
    assert mid_stage["empty_ok"] is False
    assert mid_stage["results"]["completed"] == 1
    assert mid_stage["results"]["failed"] == 0
    assert mid_stage["summary"]["completed"] == 1
    assert mid_stage["sent"] is False
    assert mid_stage["completed_at"] is None

    _write_note_style_result(run_dir, "idx-2", status="completed")
    completed_entries = [
        {"account_id": "idx-1", "status": "completed"},
        {"account_id": "idx-2", "status": "completed"},
    ]
    _write_note_style_index(run_dir, completed_entries)

    updated_final, promoted_final, log_context_final = (
        runflow_decider._apply_note_style_stage_promotion(data, run_dir)
    )

    assert updated_final is True
    assert promoted_final is True
    assert log_context_final == {"total": 2, "completed": 2, "failed": 0}

    final_stage = data["stages"]["note_style"]
    assert final_stage["status"] == "success"
    assert final_stage["empty_ok"] is False
    assert final_stage["results"]["completed"] == 2
    assert final_stage["results"]["failed"] == 0
    assert final_stage["summary"]["completed"] == 2
    assert final_stage["summary"]["empty_ok"] is False
    assert final_stage["sent"] is True
    assert isinstance(final_stage["completed_at"], str)
    assert final_stage["completed_at"].endswith("Z")


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
    assert stage_payload["status"] == "empty"
    assert stage_payload["empty_ok"] is True
    assert stage_payload["metrics"] == {"packs_total": 0}
    assert stage_payload["results"] == {"results_total": 0, "completed": 0, "failed": 0}
    assert stage_payload["summary"]["empty_ok"] is True
    assert stage_payload["summary"]["results_total"] == 0
    assert stage_payload["summary"]["completed"] == 0
    assert stage_payload["sent"] is True
    assert isinstance(stage_payload["completed_at"], str)
    assert stage_payload["completed_at"].endswith("Z")


def test_note_style_stage_promotion_partial_failures_mark_success(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    sid = "SID-note-failure"
    run_dir = runs_root / sid
    run_dir.mkdir(parents=True, exist_ok=True)

    entries = [
        {"account_id": "idx-1", "status": "completed"},
        {"account_id": "idx-2", "status": "failed"},
    ]
    _write_note_style_index(run_dir, entries)
    _write_note_style_pack(run_dir, "idx-1")
    _write_note_style_pack(run_dir, "idx-2")
    _write_note_style_result(run_dir, "idx-1", status="completed")
    _write_note_style_result(run_dir, "idx-2", status="failed")

    data: dict[str, object] = {"sid": sid, "stages": {}}
    updated, promoted, log_context = runflow_decider._apply_note_style_stage_promotion(
        data, run_dir
    )

    assert updated is True
    assert promoted is True
    assert log_context == {"total": 2, "completed": 1, "failed": 1}

    stage_payload = data["stages"]["note_style"]
    assert stage_payload["status"] == "error"
    assert stage_payload["empty_ok"] is False
    assert stage_payload["metrics"] == {"packs_total": 2}
    assert stage_payload["results"] == {"results_total": 2, "completed": 1, "failed": 1}
    assert stage_payload["summary"]["failed"] == 1
    assert stage_payload["sent"] is False
    assert stage_payload["completed_at"] is None


def test_note_style_stage_promotion_all_failed_marks_success(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    sid = "SID-note-all-failed"
    run_dir = runs_root / sid
    run_dir.mkdir(parents=True, exist_ok=True)

    entries = [
        {"account_id": "idx-1", "status": "failed"},
        {"account_id": "idx-2", "status": "failed"},
    ]
    _write_note_style_index(run_dir, entries)
    _write_note_style_pack(run_dir, "idx-1")
    _write_note_style_pack(run_dir, "idx-2")
    _write_note_style_result(run_dir, "idx-1", status="failed")
    _write_note_style_result(run_dir, "idx-2", status="failed")

    data: dict[str, object] = {"sid": sid, "stages": {}}
    updated, promoted, log_context = runflow_decider._apply_note_style_stage_promotion(
        data, run_dir
    )

    assert updated is True
    assert promoted is True
    assert log_context == {"total": 2, "completed": 0, "failed": 2}

    stage_payload = data["stages"]["note_style"]
    assert stage_payload["status"] == "error"
    assert stage_payload["empty_ok"] is False
    assert stage_payload["metrics"] == {"packs_total": 2}
    assert stage_payload["results"] == {"results_total": 2, "completed": 0, "failed": 2}
    assert stage_payload["summary"]["failed"] == 2
    assert stage_payload["sent"] is False
    assert stage_payload["completed_at"] is None


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

    _write_note_style_pack(run_dir, "idx-1")
    _write_note_style_result(run_dir, "idx-1", status="completed")

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

    _write_note_style_pack(run_dir, "idx-9")
    _write_note_style_result(
        run_dir,
        "idx-9",
        status="completed",
    )

    paths = ensure_note_style_paths(run_dir.parent, run_dir.name, create=False)
    assert paths.results_dir == custom_results.resolve()

    counts = note_style_stage_counts(run_dir)
    assert counts == {"packs_total": 1, "packs_completed": 1, "packs_failed": 0}
