from __future__ import annotations

import importlib
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace


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

import backend.core.runflow as runflow_module
import backend.runflow.decider as runflow_decider


def _load_runflow(tmp_root: Path, sid: str) -> dict:
    path = tmp_root / sid / "runflow.json"
    return json.loads(path.read_text(encoding="utf-8"))


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
