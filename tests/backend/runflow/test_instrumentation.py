from __future__ import annotations

import importlib
import json
from pathlib import Path

import backend.core.runflow as runflow


def _reload_runflow() -> None:
    importlib.reload(runflow)


def test_runflow_steps_and_events(tmp_path, monkeypatch):
    sid = "SID123"
    stage = "merge"

    monkeypatch.setenv("RUNS_ROOT", str(tmp_path))
    _reload_runflow()

    try:
        runflow.runflow_start_stage(sid, stage)
        runflow.runflow_step(
            sid,
            stage,
            "load_cases",
            metrics={"accounts": 3},
        )

        # Second start is idempotent and should not clear existing steps.
        runflow.runflow_start_stage(sid, stage)

        runflow.runflow_step(
            sid,
            stage,
            "load_cases",
            metrics={"accounts": 4},
        )
        runflow.runflow_step(
            sid,
            stage,
            "score_pairs",
            account="acct-1",
            metrics={"pairs": 10},
            out={"path": "merge/score.json"},
        )
        runflow.runflow_end_stage(
            sid,
            stage,
            status="success",
            summary={"accounts_seen": 4},
        )

        steps_path = Path(tmp_path, sid, "runflow_steps.json")
        events_path = Path(tmp_path, sid, "runflow_events.jsonl")

        assert steps_path.exists()
        assert events_path.exists()

        steps_payload = json.loads(steps_path.read_text(encoding="utf-8"))
        assert steps_payload["sid"] == sid
        assert steps_payload["schema_version"] == "2.0"

        merge_stage = steps_payload["stages"][stage]
        assert merge_stage["status"] == "success"
        assert "started_at" in merge_stage
        assert "ended_at" in merge_stage
        assert merge_stage["summary"]["accounts_seen"] == 4

        steps = merge_stage["steps"]
        assert len(steps) == 2

        load_cases_entry = next(item for item in steps if item["name"] == "load_cases")
        assert load_cases_entry["status"] == "success"
        assert load_cases_entry["metrics"] == {"accounts": 4}

        score_entry = next(item for item in steps if item["name"] == "score_pairs")
        assert score_entry["account"] == "acct-1"
        assert score_entry["metrics"] == {"pairs": 10}
        assert score_entry["out"] == {"path": "merge/score.json"}

        # Events log should include one start, per-step events, and the end event.
        events = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()]
        start_events = [event for event in events if event.get("event") == "start"]
        assert len(start_events) == 1
        assert start_events[0]["stage"] == stage

        end_events = [event for event in events if event.get("event") == "end"]
        assert len(end_events) == 1
        assert end_events[0]["status"] == "success"
        assert end_events[0]["summary"] == {"accounts_seen": 4}

        step_events = [event for event in events if event.get("step") == "load_cases"]
        # load_cases is emitted twice, once per update
        assert len(step_events) == 2
        assert step_events[-1]["metrics"] == {"accounts": 4}
    finally:
        monkeypatch.delenv("RUNS_ROOT", raising=False)
        _reload_runflow()
