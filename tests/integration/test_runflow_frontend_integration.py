from __future__ import annotations

import importlib
import json
from pathlib import Path

import backend.core.runflow as runflow_module
import backend.frontend.packs.generator as frontend_generator
import backend.runflow.decider as runflow_decider


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_runflow_completes_when_validation_has_no_findings(tmp_path):
    runs_root = tmp_path / "runs"
    sid = "SID-no-findings"

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
    assert decision == {"next": "complete_no_action", "reason": "validation_no_findings"}

    result = frontend_generator.generate_frontend_packs_for_run(
        sid, runs_root=runs_root
    )
    assert result["packs_count"] == 0

    runflow_decider.record_stage(
        sid,
        "frontend",
        status="published",
        counts={"packs_count": result["packs_count"]},
        empty_ok=True,
        runs_root=runs_root,
    )

    follow_up = runflow_decider.decide_next(sid, runs_root=runs_root)
    assert follow_up["next"] == "complete_no_action"
    assert follow_up["reason"] in {"validation_no_findings", "frontend_no_packs"}

    runflow_data = json.loads(
        (runs_root / sid / "runflow.json").read_text(encoding="utf-8")
    )
    assert runflow_data["run_state"] == "COMPLETE_NO_ACTION"
    assert runflow_data["stages"]["frontend"]["packs_count"] == 0

    index_path = runs_root / sid / "frontend" / "review" / "index.json"
    assert index_path.exists()
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    assert payload["packs"] == []


def test_runflow_generates_frontend_and_moves_to_await(tmp_path):
    runs_root = tmp_path / "runs"
    sid = "SID-with-findings"

    # Prepare two accounts for the generator.
    for account_id in ("acct-1", "acct-2"):
        account_dir = runs_root / sid / "cases" / "accounts" / account_id
        _write_json(
            account_dir / "summary.json",
            {
                "account_id": account_id,
                "labels": {"creditor": f"Creditor {account_id}"},
            },
        )
        _write_json(
            account_dir / "bureaus.json",
            {
                "transunion": {
                    "account_number_display": f"****{account_id[-1]}001",
                    "account_type": "Credit Card",
                    "account_status": "Open",
                }
            },
        )

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
        counts={"findings_count": 2},
        empty_ok=False,
        runs_root=runs_root,
    )

    decision = runflow_decider.decide_next(sid, runs_root=runs_root)
    assert decision == {"next": "gen_frontend_packs", "reason": "validation_has_findings"}

    result = frontend_generator.generate_frontend_packs_for_run(
        sid, runs_root=runs_root
    )
    assert result["packs_count"] == 2

    runflow_decider.record_stage(
        sid,
        "frontend",
        status="published",
        counts={"packs_count": 2},
        empty_ok=False,
        runs_root=runs_root,
    )

    follow_up = runflow_decider.decide_next(sid, runs_root=runs_root)
    assert follow_up == {"next": "await_input", "reason": "frontend_published"}

    index_path = runs_root / sid / "frontend" / "review" / "index.json"
    assert index_path.exists()
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    assert len(payload["packs"]) == 2


def test_runflow_instrumentation_smoke(tmp_path, monkeypatch):
    runs_root = tmp_path / "runs"
    sid = "SID-instrumented"

    monkeypatch.setenv("RUNS_ROOT", str(runs_root))
    monkeypatch.setenv("RUNFLOW_VERBOSE", "1")
    monkeypatch.setenv("RUNFLOW_EVENTS", "1")
    monkeypatch.setenv("RUNFLOW_STEP_LOG_EVERY", "1")

    importlib.reload(runflow_module)
    importlib.reload(runflow_decider)
    importlib.reload(frontend_generator)

    try:
        runflow_module.runflow_start_stage(sid, "merge")
        runflow_module.runflow_step(
            sid,
            "merge",
            "load_cases",
            metrics={"accounts": 1},
        )
        runflow_module.runflow_step(
            sid,
            "merge",
            "load_cases",
            metrics={"accounts": 2},
        )
        runflow_module.runflow_step(
            sid,
            "merge",
            "score_pairs",
            metrics={"pairs": 3},
        )
        runflow_module.runflow_end_stage(
            sid,
            "merge",
            summary={"accounts_seen": 2},
        )

        runflow_decider.record_stage(
            sid,
            "merge",
            status="success",
            counts={"count": 2},
            empty_ok=False,
            runs_root=runs_root,
        )

        runflow_module.runflow_start_stage(sid, "validation")
        runflow_module.runflow_step(
            sid,
            "validation",
            "apply_rules",
            metrics={"findings": 1},
        )
        runflow_module.runflow_step(
            sid,
            "validation",
            "apply_rules",
            metrics={"findings": 1},
        )
        runflow_module.runflow_end_stage(
            sid,
            "validation",
            summary={"findings_count": 1},
        )

        runflow_decider.record_stage(
            sid,
            "validation",
            status="built",
            counts={"findings_count": 1},
            empty_ok=False,
            runs_root=runs_root,
        )

        # Prepare minimal account data for the frontend generator.
        for account_id in ("acct-1", "acct-2"):
            account_dir = runs_root / sid / "cases" / "accounts" / account_id
            account_dir.mkdir(parents=True, exist_ok=True)
            _write_json(
                account_dir / "summary.json",
                {
                    "account_id": account_id,
                    "labels": {"creditor": f"Creditor {account_id}"},
                },
            )
            _write_json(
                account_dir / "bureaus.json",
                {
                    "transunion": {
                        "account_number_display": f"****{account_id[-1]}001",
                        "account_type": "Credit Card",
                        "account_status": "Open",
                    }
                },
            )

        frontend_result = frontend_generator.generate_frontend_packs_for_run(
            sid, runs_root=runs_root
        )

        runflow_decider.record_stage(
            sid,
            "frontend",
            status="published",
            counts={"packs_count": frontend_result["packs_count"]},
            empty_ok=False,
            runs_root=runs_root,
        )

        runflow_data = json.loads(
            (runs_root / sid / "runflow.json").read_text(encoding="utf-8")
        )
        assert runflow_data["stages"]["merge"]["count"] == 2
        assert runflow_data["stages"]["validation"]["findings_count"] == 1
        frontend_stage = runflow_data["stages"]["frontend"]
        assert frontend_stage["packs_count"] == frontend_result["packs_count"]
        metrics_payload = frontend_stage.get("metrics") or {}
        assert metrics_payload.get("answers_required") == frontend_result["packs_count"]
        assert metrics_payload.get("answers_received") == 0

        steps_path = runs_root / sid / "runflow_steps.json"
        events_path = runs_root / sid / "runflow_events.jsonl"

        assert steps_path.exists()
        assert events_path.exists()

        steps_payload = json.loads(steps_path.read_text(encoding="utf-8"))
        events_lines = events_path.read_text(encoding="utf-8").splitlines()
        events_payload = [json.loads(line) for line in events_lines if line.strip()]

        merge_steps = steps_payload["stages"]["merge"]["substages"]["default"]["steps"]
        if merge_steps:
            assert len(merge_steps) == 2
            load_cases_entry = next(
                item for item in merge_steps if item["name"] == "load_cases"
            )
            assert load_cases_entry["metrics"] == {"accounts": 2}
        else:
            merge_events = [
                entry
                for entry in events_payload
                if entry.get("stage") == "merge" and entry.get("step") == "load_cases"
            ]
            assert merge_events, "expected merge load_cases events"
            assert merge_events[-1].get("metrics") == {"accounts": 2}

        validation_steps = steps_payload["stages"]["validation"]["substages"]["default"]["steps"]
        assert len(validation_steps) == 1
        assert validation_steps[0]["metrics"] == {"findings": 1}

        frontend_steps = steps_payload["stages"]["frontend"]["steps"]
        step_names = [entry["name"] for entry in frontend_steps]
        assert step_names[0] == "frontend_review_start"
        assert "frontend_review_finish" in step_names
        responses_entry = next(
            entry for entry in frontend_steps if entry["name"] == "responses_progress"
        )
        assert responses_entry["metrics"] == {
            "accounts_published": frontend_result["packs_count"],
            "answers_received": 0,
            "answers_required": frontend_result["packs_count"],
        }

        assert len(events_lines) >= 6

        # Ensure the append-only log retains the first event when more events are added.
        first_event = events_payload[0]
        assert first_event["event"] == "start"
        assert first_event["stage"] == "merge"
    finally:
        monkeypatch.delenv("RUNS_ROOT", raising=False)
        monkeypatch.delenv("RUNFLOW_VERBOSE", raising=False)
        monkeypatch.delenv("RUNFLOW_EVENTS", raising=False)
        monkeypatch.delenv("RUNFLOW_STEP_LOG_EVERY", raising=False)
        importlib.reload(runflow_module)
        importlib.reload(runflow_decider)
        importlib.reload(frontend_generator)
