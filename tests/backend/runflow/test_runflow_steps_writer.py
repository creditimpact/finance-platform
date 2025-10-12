from __future__ import annotations

import importlib
import json
from datetime import datetime
from pathlib import Path


def _reload_steps_module():
    import backend.core.runflow_steps as runflow_steps

    importlib.reload(runflow_steps)
    return runflow_steps


def _load_steps_payload(runs_root: Path, sid: str) -> dict:
    path = runs_root / sid / "runflow_steps.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_steps_writer_seq_monotonic_and_next_seq(tmp_path, monkeypatch):
    runs_root = tmp_path / "runs"
    sid = "SID-seq"
    stage = "merge"

    monkeypatch.setenv("RUNS_ROOT", str(runs_root))
    runflow_steps = _reload_steps_module()

    runflow_steps.steps_stage_start(sid, stage, started_at="2024-01-01T00:00:00Z")

    runflow_steps.steps_append(
        sid,
        stage,
        "first",
        "success",
        seq=5,
        t="2024-01-01T00:00:01Z",
        metrics={"value": 1},
    )
    runflow_steps.steps_append(
        sid,
        stage,
        "second",
        "success",
        seq=2,
        t="2024-01-01T00:00:02Z",
    )
    runflow_steps.steps_append(
        sid,
        stage,
        "third",
        "success",
        seq=None,
        t="2024-01-01T00:00:03Z",
    )

    payload = _load_steps_payload(runs_root, sid)
    stage_payload = payload["stages"][stage]

    seqs = [entry["seq"] for entry in stage_payload["steps"]]
    assert seqs == sorted(seqs)
    assert seqs == sorted(set(seqs))
    assert stage_payload["next_seq"] == max(seqs) + 1


def test_steps_writer_updated_at_tracks_stage_timestamps(tmp_path, monkeypatch):
    runs_root = tmp_path / "runs"
    sid = "SID-updated"
    stage = "validation"

    monkeypatch.setenv("RUNS_ROOT", str(runs_root))
    runflow_steps = _reload_steps_module()

    runflow_steps.steps_stage_start(sid, stage, started_at="2024-02-01T00:00:00Z")
    runflow_steps.steps_append(
        sid,
        stage,
        "rule_apply",
        "success",
        t="2024-02-01T00:00:01Z",
    )
    runflow_steps.steps_stage_finish(
        sid,
        stage,
        "success",
        summary={"findings": 0},
        ended_at="2024-02-01T00:00:05Z",
    )

    payload = _load_steps_payload(runs_root, sid)
    updated_at = payload["updated_at"]
    assert updated_at.endswith("Z")

    ended_at = payload["stages"][stage]["ended_at"]
    assert ended_at == "2024-02-01T00:00:05Z"

    # ``updated_at`` should be the most recent timestamp and no earlier than ``ended_at``.
    def _parse(value: str) -> float:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()

    assert _parse(updated_at) >= _parse(ended_at)


def test_steps_writer_marks_empty_ok(tmp_path, monkeypatch):
    runs_root = tmp_path / "runs"
    sid = "SID-empty"
    stage = "merge"

    monkeypatch.setenv("RUNS_ROOT", str(runs_root))
    runflow_steps = _reload_steps_module()

    runflow_steps.steps_stage_start(sid, stage, started_at="2024-03-01T00:00:00Z")
    runflow_steps.steps_stage_finish(
        sid,
        stage,
        "empty",
        summary={"scored_pairs": 0, "empty_ok": True},
        ended_at="2024-03-01T00:00:02Z",
        empty_ok=True,
    )

    payload = _load_steps_payload(runs_root, sid)
    stage_payload = payload["stages"][stage]

    assert stage_payload["status"] == "empty"
    assert stage_payload["empty_ok"] is True
    assert stage_payload["steps"] == []
    default_substage = stage_payload["substages"]["default"]
    assert default_substage["steps"] == []

