from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

import backend.core.runflow as runflow
import backend.core.runflow_steps as runflow_steps


def _reload_runflow() -> None:
    importlib.reload(runflow_steps)
    importlib.reload(runflow)


def test_runflow_steps_and_events(tmp_path, monkeypatch):
    sid = "SID123"
    stage = "merge"

    monkeypatch.setenv("RUNS_ROOT", str(tmp_path))
    monkeypatch.setenv("RUNFLOW_VERBOSE", "1")
    monkeypatch.setenv("RUNFLOW_EVENTS", "1")
    monkeypatch.setenv("RUNFLOW_STEP_LOG_EVERY", "1")
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
        assert steps_payload["schema_version"] == "2.1"

        merge_stage = steps_payload["stages"][stage]
        assert merge_stage["status"] == "success"
        assert "started_at" in merge_stage
        assert "ended_at" in merge_stage
        assert merge_stage["summary"]["accounts_seen"] == 4

        stage_steps = merge_stage["steps"]
        assert [entry["name"] for entry in stage_steps] == [
            "load_cases",
            "load_cases",
            "score_pairs",
        ]
        assert [entry["seq"] for entry in stage_steps] == [1, 2, 3]
        assert merge_stage["next_seq"] == 4
        assert stage_steps[1]["metrics"] == {"accounts": 4}

        substages = merge_stage["substages"]
        assert set(substages.keys()) == {"default"}

        default_substage = substages["default"]
        assert default_substage["status"] == "success"
        assert "started_at" in default_substage
        steps = default_substage["steps"]
        assert len(steps) == 2

        load_cases_entry = next(item for item in steps if item["name"] == "load_cases")
        assert load_cases_entry["status"] == "success"
        assert load_cases_entry["metrics"] == {"accounts": 4}
        assert load_cases_entry["seq"] == 2

        score_entry = next(item for item in steps if item["name"] == "score_pairs")
        assert score_entry["account"] == "acct-1"
        assert score_entry["metrics"] == {"pairs": 10}
        assert score_entry["out"] == {"path": "merge/score.json"}
        assert score_entry["seq"] == 3

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
        assert step_events[-1]["substage"] == "default"
    finally:
        monkeypatch.delenv("RUNS_ROOT", raising=False)
        monkeypatch.delenv("RUNFLOW_VERBOSE", raising=False)
        monkeypatch.delenv("RUNFLOW_EVENTS", raising=False)
        monkeypatch.delenv("RUNFLOW_STEP_LOG_EVERY", raising=False)
        _reload_runflow()


def test_runflow_step_sampling(tmp_path, monkeypatch):
    sid = "SID-SAMPLE"
    stage = "validation"

    monkeypatch.setenv("RUNS_ROOT", str(tmp_path))
    monkeypatch.setenv("RUNFLOW_VERBOSE", "1")
    monkeypatch.setenv("RUNFLOW_EVENTS", "1")
    monkeypatch.setenv("RUNFLOW_STEP_LOG_EVERY", "3")
    _reload_runflow()

    try:
        runflow.runflow_start_stage(sid, stage)
        for idx in range(5):
            runflow.runflow_step(
                sid,
                stage,
                "evaluate",
                metrics={"iteration": idx},
            )

        steps_path = Path(tmp_path, sid, "runflow_steps.json")
        events_path = Path(tmp_path, sid, "runflow_events.jsonl")

        steps_payload = json.loads(steps_path.read_text(encoding="utf-8"))
        stage_payload = steps_payload["stages"][stage]
        stage_steps = stage_payload["steps"]
        assert [entry["metrics"] for entry in stage_steps] == [
            {"iteration": 0},
            {"iteration": 2},
        ]

        default_substage = stage_payload["substages"]["default"]
        steps = default_substage["steps"]
        assert len(steps) == 1
        assert steps[0]["metrics"] == {"iteration": 2}

        events = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()]
        evaluate_events = [event for event in events if event.get("step") == "evaluate"]
        assert len(evaluate_events) == 2
        assert evaluate_events[-1]["metrics"] == {"iteration": 2}
        assert all(event["substage"] == "default" for event in evaluate_events)
    finally:
        monkeypatch.delenv("RUNS_ROOT", raising=False)
        monkeypatch.delenv("RUNFLOW_VERBOSE", raising=False)
        monkeypatch.delenv("RUNFLOW_EVENTS", raising=False)
        monkeypatch.delenv("RUNFLOW_STEP_LOG_EVERY", raising=False)
        _reload_runflow()


def test_runflow_step_dec_error_records_and_reraises(tmp_path, monkeypatch):
    sid = "SID-DECORATOR"
    stage = "merge"

    monkeypatch.setenv("RUNS_ROOT", str(tmp_path))
    monkeypatch.setenv("RUNFLOW_VERBOSE", "1")
    monkeypatch.setenv("RUNFLOW_EVENTS", "1")
    monkeypatch.setenv("RUNFLOW_STEP_LOG_EVERY", "1")
    _reload_runflow()

    try:
        runflow.runflow_start_stage(sid, stage)

        @runflow.runflow_step_dec(stage, "decorated_step")
        def _boom(payload):
            raise RuntimeError("boom")

        with pytest.raises(RuntimeError):
            _boom({"sid": sid})

        steps_path = Path(tmp_path, sid, "runflow_steps.json")
        events_path = Path(tmp_path, sid, "runflow_events.jsonl")

        steps_payload = json.loads(steps_path.read_text(encoding="utf-8"))
        stage_payload = steps_payload["stages"][stage]
        assert stage_payload["status"] == "running"
        assert "ended_at" not in stage_payload

        default_substage = stage_payload["substages"]["default"]
        assert default_substage["status"] == "error"
        step_entries = default_substage["steps"]
        assert len(step_entries) == 1

        entry = step_entries[0]
        assert entry["name"] == "decorated_step"
        assert entry["status"] == "error"
        assert entry["out"] == {"error": "RuntimeError", "msg": "boom"}

        stage_steps = stage_payload["steps"]
        assert len(stage_steps) == 1
        assert stage_steps[0]["status"] == "error"

        events = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()]
        end_events = [event for event in events if event.get("event") == "end"]
        assert not end_events

        step_events = [event for event in events if event.get("step") == "decorated_step"]
        assert len(step_events) == 1
        assert step_events[0]["status"] == "error"
        assert step_events[0]["out"] == {"error": "RuntimeError", "msg": "boom"}
    finally:
        monkeypatch.delenv("RUNS_ROOT", raising=False)
        monkeypatch.delenv("RUNFLOW_VERBOSE", raising=False)
        monkeypatch.delenv("RUNFLOW_EVENTS", raising=False)
        monkeypatch.delenv("RUNFLOW_STEP_LOG_EVERY", raising=False)
        _reload_runflow()


def test_runflow_end_stage_records_empty_status(tmp_path, monkeypatch):
    sid = "SID-empty"
    stage = "frontend"

    monkeypatch.setenv("RUNS_ROOT", str(tmp_path))
    monkeypatch.setenv("RUNFLOW_VERBOSE", "1")
    _reload_runflow()

    try:
        runflow.runflow_start_stage(sid, stage)
        runflow.runflow_end_stage(
            sid,
            stage,
            summary={"packs_count": 0},
            stage_status="empty",
            empty_ok=True,
        )

        steps_path = Path(tmp_path, sid, "runflow_steps.json")
        steps_payload = json.loads(steps_path.read_text(encoding="utf-8"))
        stage_payload = steps_payload["stages"][stage]
        assert stage_payload["status"] == "empty"
        assert stage_payload.get("empty_ok") is True
    finally:
        monkeypatch.delenv("RUNS_ROOT", raising=False)
        monkeypatch.delenv("RUNFLOW_VERBOSE", raising=False)
        _reload_runflow()


def test_runflow_atomic_writes_and_event_appends(tmp_path, monkeypatch):
    sid = "SID-ATOMIC"
    stage = "merge"

    monkeypatch.setenv("RUNS_ROOT", str(tmp_path))
    monkeypatch.setenv("RUNFLOW_VERBOSE", "1")
    monkeypatch.setenv("RUNFLOW_EVENTS", "1")
    monkeypatch.setenv("RUNFLOW_STEP_LOG_EVERY", "1")
    _reload_runflow()

    try:
        original_atomic = runflow_steps._atomic_write_json
        calls: list[Path] = []

        def _tracking_atomic(path: Path, payload):
            calls.append(Path(path))
            original_atomic(path, payload)

        monkeypatch.setattr(runflow_steps, "_atomic_write_json", _tracking_atomic)

        uuid_values = iter(["aa", "bb", "cc", "dd"])

        class _FakeUUID:
            def __init__(self, hex_value: str) -> None:
                self.hex = hex_value

        monkeypatch.setattr(
            runflow_steps.uuid,
            "uuid4",
            lambda: _FakeUUID(next(uuid_values)),
        )

        run_dir = Path(tmp_path, sid)
        steps_path = run_dir / "runflow_steps.json"
        tmp_paths = [
            steps_path.with_suffix(steps_path.suffix + ".tmp.aa"),
            steps_path.with_suffix(steps_path.suffix + ".tmp.bb"),
            steps_path.with_suffix(steps_path.suffix + ".tmp.cc"),
            steps_path.with_suffix(steps_path.suffix + ".tmp.dd"),
        ]

        runflow.runflow_start_stage(sid, stage)
        runflow.runflow_step(sid, stage, "load_cases", metrics={"accounts": 1})
        runflow.runflow_end_stage(sid, stage, summary={"accounts_seen": 1})

        # Ensure atomic writes were triggered for init, start, step, and end operations.
        assert calls == [steps_path, steps_path, steps_path, steps_path]

        for path in tmp_paths:
            assert not path.exists()

        events_path = run_dir / "runflow_events.jsonl"
        assert steps_path.exists()
        assert events_path.exists()

        lines = events_path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 3

        first_event = json.loads(lines[0])
        assert first_event["event"] == "start"
        assert first_event["stage"] == stage
        assert set(first_event) == {"ts", "stage", "event"}

        step_event = json.loads(lines[1])
        assert step_event["step"] == "load_cases"
        assert step_event["metrics"] == {"accounts": 1}
        assert step_event["substage"] == "default"

        end_event = json.loads(lines[2])
        assert end_event["event"] == "end"
        assert end_event["summary"] == {"accounts_seen": 1}
        assert end_event["status"] == "success"
    finally:
        monkeypatch.delenv("RUNS_ROOT", raising=False)
        monkeypatch.delenv("RUNFLOW_VERBOSE", raising=False)
        monkeypatch.delenv("RUNFLOW_EVENTS", raising=False)
        monkeypatch.delenv("RUNFLOW_STEP_LOG_EVERY", raising=False)
        _reload_runflow()
