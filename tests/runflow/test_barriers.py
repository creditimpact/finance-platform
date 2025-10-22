from __future__ import annotations

import copy
import importlib
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import backend.core.runflow as runflow
import backend.core.runflow_steps as runflow_steps


def _reload_runflow() -> None:
    importlib.reload(runflow_steps)
    importlib.reload(runflow)


def _ensure_requests_stub(monkeypatch) -> None:
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
    monkeypatch.setitem(sys.modules, "requests", module)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _prepare_runflow_files(run_dir: Path, *, stages: dict[str, dict], runflow_payload: dict | None = None) -> None:
    _write_json(run_dir / "runflow_steps.json", {"stages": stages})
    stage_snapshot = {key: copy.deepcopy(value) for key, value in stages.items()}
    default_payload = {"sid": run_dir.name, "note": "persist"}
    if runflow_payload is not None:
        payload = copy.deepcopy(runflow_payload)
        if not isinstance(payload, dict):
            payload = dict(default_payload)
    else:
        payload = dict(default_payload)

    payload.setdefault("sid", run_dir.name)
    existing_stages = payload.get("stages")
    if isinstance(existing_stages, dict):
        combined_stages = copy.deepcopy(existing_stages)
        combined_stages.update(stage_snapshot)
    else:
        combined_stages = stage_snapshot

    payload["stages"] = combined_stages
    _write_json(run_dir / "runflow.json", payload)


def _prepare_validation_artifacts(run_dir: Path, *, sid: str, account_number: int = 1) -> None:
    validation_dir = run_dir / "ai_packs" / "validation"
    results_dir = validation_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    result_filename = f"idx-{account_number:03d}.result.jsonl"
    (results_dir / result_filename).write_text("{}", encoding="utf-8")

    index_payload = {
        "schema_version": 2,
        "sid": sid,
        "root": ".",
        "packs_dir": "packs",
        "results_dir": "results",
        "packs": [
            {
                "account_id": account_number,
                "pack": f"packs/idx-{account_number:03d}.json",
                "result_json": f"results/{result_filename}",
                "status": "completed",
                "lines": 0,
            }
        ],
    }

    _write_json(validation_dir / "index.json", index_payload)


def _prepare_review_manifest(run_dir: Path, account_id: str) -> None:
    review_dir = run_dir / "frontend" / "review"
    review_dir.mkdir(parents=True, exist_ok=True)
    _write_json(review_dir / "index.json", {"packs": [{"account_id": account_id}]})


def _write_review_response(run_dir: Path, account_id: str) -> None:
    responses_dir = run_dir / "frontend" / "review" / "responses"
    responses_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "received_at": "2024-01-01T00:00:00Z",
        "answers": {"explanation": "ready"},
    }
    from backend.runflow.decider import _response_filename_for_account

    filename = _response_filename_for_account(account_id)
    _write_json(responses_dir / filename, payload)


def _load_runflow_payload(run_dir: Path) -> dict:
    return json.loads((run_dir / "runflow.json").read_text(encoding="utf-8"))


def test_merge_stage_sets_only_merge_ready(tmp_path, monkeypatch):
    sid = "merge-only"
    monkeypatch.setenv("RUNS_ROOT", str(tmp_path))
    monkeypatch.setenv("UMBRELLA_BARRIERS_ENABLED", "1")
    monkeypatch.setenv("UMBRELLA_BARRIERS_LOG", "0")
    _ensure_requests_stub(monkeypatch)
    _reload_runflow()

    try:
        run_dir = tmp_path / sid
        stages = {
            "merge": {
                "status": "success",
                "summary": {"result_files": 1},
            }
        }
        _prepare_runflow_files(run_dir, stages=stages)

        before = {entry.name for entry in run_dir.iterdir()}
        runflow.runflow_barriers_refresh(sid)
        after = {entry.name for entry in run_dir.iterdir()}

        assert before == after

        payload = _load_runflow_payload(run_dir)
        assert payload["note"] == "persist"
        umbrella = payload["umbrella_barriers"]
        assert umbrella["merge_ready"] is True
        assert umbrella["validation_ready"] is False
        assert umbrella["review_ready"] is False
        assert umbrella["all_ready"] is False
        assert payload["umbrella_ready"] is False
        assert isinstance(umbrella["checked_at"], str)
    finally:
        monkeypatch.delenv("RUNS_ROOT", raising=False)
        monkeypatch.delenv("UMBRELLA_BARRIERS_ENABLED", raising=False)
        monkeypatch.delenv("UMBRELLA_BARRIERS_LOG", raising=False)
        _reload_runflow()


def test_record_stage_updates_barriers_without_instrumentation(tmp_path, monkeypatch):
    sid = "merge-record"
    monkeypatch.setenv("RUNS_ROOT", str(tmp_path))
    monkeypatch.setenv("UMBRELLA_BARRIERS_ENABLED", "1")
    monkeypatch.setenv("UMBRELLA_BARRIERS_LOG", "0")
    monkeypatch.delenv("RUNFLOW_VERBOSE", raising=False)
    monkeypatch.delenv("RUNFLOW_EVENTS", raising=False)
    _reload_runflow()

    try:
        import backend.runflow.decider as decider

        importlib.reload(decider)

        decider.record_stage(
            sid,
            "merge",
            status="success",
            counts={
                "pairs_scored": 1,
                "packs_created": 1,
                "result_files": 1,
            },
            empty_ok=False,
            runs_root=tmp_path,
        )

        run_dir = tmp_path / sid
        payload = _load_runflow_payload(run_dir)
        umbrella = payload["umbrella_barriers"]

        assert umbrella["merge_ready"] is True
        assert umbrella["validation_ready"] is False
        assert umbrella["review_ready"] is False
        assert payload["umbrella_ready"] is False
        merge_stage = payload["stages"]["merge"]
        summary = merge_stage["summary"]
        assert summary["pairs_scored"] == 1
        assert summary["packs_created"] == 1
        assert summary["result_files"] == 1
        assert not (run_dir / "runflow_steps.json").exists()
    finally:
        monkeypatch.delenv("RUNS_ROOT", raising=False)
        monkeypatch.delenv("UMBRELLA_BARRIERS_ENABLED", raising=False)
        monkeypatch.delenv("UMBRELLA_BARRIERS_LOG", raising=False)
        monkeypatch.delenv("RUNFLOW_VERBOSE", raising=False)
        monkeypatch.delenv("RUNFLOW_EVENTS", raising=False)
        _reload_runflow()


def test_reconcile_barriers_honors_legacy_result_files(tmp_path, monkeypatch):
    sid = "merge-legacy"
    monkeypatch.setenv("RUNS_ROOT", str(tmp_path))
    monkeypatch.setenv("UMBRELLA_BARRIERS_ENABLED", "1")
    monkeypatch.setenv("UMBRELLA_BARRIERS_LOG", "0")
    _reload_runflow()

    try:
        run_dir = tmp_path / sid
        stages = {
            "merge": {
                "status": "success",
                "result_files": 2,
            }
        }
        _prepare_runflow_files(run_dir, stages=stages)

        import backend.runflow.decider as decider

        importlib.reload(decider)

        decider.reconcile_umbrella_barriers(sid, runs_root=tmp_path)

        payload = _load_runflow_payload(run_dir)
        umbrella = payload["umbrella_barriers"]

        assert umbrella["merge_ready"] is True
        assert umbrella["validation_ready"] is False
        assert umbrella["review_ready"] is False
        assert umbrella["all_ready"] is False
        assert payload["umbrella_ready"] is False
    finally:
        monkeypatch.delenv("RUNS_ROOT", raising=False)
        monkeypatch.delenv("UMBRELLA_BARRIERS_ENABLED", raising=False)
        monkeypatch.delenv("UMBRELLA_BARRIERS_LOG", raising=False)
        _reload_runflow()


def test_validation_stage_sets_validation_ready(tmp_path, monkeypatch):
    sid = "validation-only"
    monkeypatch.setenv("RUNS_ROOT", str(tmp_path))
    monkeypatch.setenv("UMBRELLA_BARRIERS_ENABLED", "1")
    monkeypatch.setenv("UMBRELLA_BARRIERS_LOG", "0")
    _ensure_requests_stub(monkeypatch)
    _reload_runflow()

    try:
        run_dir = tmp_path / sid
        stages = {
            "validation": {
                "status": "success",
                "results": {"results_total": 1, "completed": 1},
            }
        }
        _prepare_runflow_files(run_dir, stages=stages)
        _prepare_validation_artifacts(run_dir, sid=sid, account_number=42)

        runflow.runflow_barriers_refresh(sid)
        payload = _load_runflow_payload(run_dir)
        umbrella = payload["umbrella_barriers"]

        assert umbrella["merge_ready"] is False
        assert umbrella["validation_ready"] is True
        assert umbrella["review_ready"] is False
        assert umbrella["all_ready"] is False
        assert payload["umbrella_ready"] is False
    finally:
        monkeypatch.delenv("RUNS_ROOT", raising=False)
        monkeypatch.delenv("UMBRELLA_BARRIERS_ENABLED", raising=False)
        monkeypatch.delenv("UMBRELLA_BARRIERS_LOG", raising=False)
        _reload_runflow()


def test_validation_readiness_respects_env_overrides(tmp_path, monkeypatch):
    sid = "validation-env-override"
    monkeypatch.setenv("RUNS_ROOT", str(tmp_path))
    monkeypatch.setenv("UMBRELLA_BARRIERS_ENABLED", "1")
    monkeypatch.setenv("UMBRELLA_BARRIERS_LOG", "0")
    monkeypatch.setenv("VALIDATION_INDEX_PATH", "external/index.json")
    monkeypatch.setenv("VALIDATION_RESULTS_DIR", "external/results")
    _ensure_requests_stub(monkeypatch)
    _reload_runflow()

    try:
        run_dir = tmp_path / sid
        stages = {
            "validation": {
                "status": "success",
                "results": {"results_total": 1, "completed": 1},
            }
        }
        _prepare_runflow_files(run_dir, stages=stages)

        external_dir = run_dir / "external"
        results_dir = external_dir / "results"
        results_dir.mkdir(parents=True, exist_ok=True)

        account_number = 5
        result_filename = f"idx-{account_number:03d}.result.jsonl"
        (results_dir / result_filename).write_text("{}", encoding="utf-8")

        index_payload = {
            "schema_version": 2,
            "sid": sid,
            "root": ".",
            "packs_dir": "packs",
            "results_dir": "results",
            "packs": [
                {
                    "account_id": account_number,
                    "pack": f"packs/idx-{account_number:03d}.json",
                    "result_json": f"results/{result_filename}",
                    "status": "completed",
                    "lines": 1,
                }
            ],
        }

        _write_json(external_dir / "index.json", index_payload)

        runflow.runflow_barriers_refresh(sid)
        payload = _load_runflow_payload(run_dir)
        umbrella = payload["umbrella_barriers"]

        assert umbrella["merge_ready"] is False
        assert umbrella["validation_ready"] is True
        assert umbrella["review_ready"] is False
        assert umbrella["all_ready"] is False
        assert payload["umbrella_ready"] is False
    finally:
        monkeypatch.delenv("RUNS_ROOT", raising=False)
        monkeypatch.delenv("UMBRELLA_BARRIERS_ENABLED", raising=False)
        monkeypatch.delenv("UMBRELLA_BARRIERS_LOG", raising=False)
        monkeypatch.delenv("VALIDATION_INDEX_PATH", raising=False)
        monkeypatch.delenv("VALIDATION_RESULTS_DIR", raising=False)
        _reload_runflow()


def test_all_stages_ready_marks_run_ready(tmp_path, monkeypatch):
    sid = "all-ready"
    monkeypatch.setenv("RUNS_ROOT", str(tmp_path))
    monkeypatch.setenv("UMBRELLA_BARRIERS_ENABLED", "1")
    monkeypatch.setenv("UMBRELLA_BARRIERS_LOG", "0")
    _ensure_requests_stub(monkeypatch)
    _reload_runflow()

    try:
        run_dir = tmp_path / sid
        stages = {
            "merge": {
                "status": "success",
                "summary": {"result_files": 1},
            },
            "validation": {
                "status": "success",
                "results": {"results_total": 1, "completed": 1},
            },
            "frontend": {
                "status": "success",
                "metrics": {"answers_required": 1, "answers_received": 1},
            },
        }
        _prepare_runflow_files(run_dir, stages=stages)
        _prepare_validation_artifacts(run_dir, sid=sid, account_number=7)

        account_id = "idx-007"
        _prepare_review_manifest(run_dir, account_id)
        _write_review_response(run_dir, account_id)

        runflow.runflow_barriers_refresh(sid)

        payload = _load_runflow_payload(run_dir)
        umbrella = payload["umbrella_barriers"]

        assert umbrella["merge_ready"] is True
        assert umbrella["validation_ready"] is True
        assert umbrella["review_ready"] is True
        assert umbrella["all_ready"] is True
        assert payload["umbrella_ready"] is True
    finally:
        monkeypatch.delenv("RUNS_ROOT", raising=False)
        monkeypatch.delenv("UMBRELLA_BARRIERS_ENABLED", raising=False)
        monkeypatch.delenv("UMBRELLA_BARRIERS_LOG", raising=False)
        _reload_runflow()


def test_review_readiness_updates_when_response_arrives(tmp_path, monkeypatch):
    sid = "awaiting-review"
    monkeypatch.setenv("RUNS_ROOT", str(tmp_path))
    monkeypatch.setenv("UMBRELLA_BARRIERS_ENABLED", "1")
    monkeypatch.setenv("UMBRELLA_BARRIERS_LOG", "0")
    _ensure_requests_stub(monkeypatch)
    _reload_runflow()

    try:
        run_dir = tmp_path / sid
        stages = {
            "merge": {
                "status": "success",
                "summary": {"result_files": 1},
            },
            "validation": {
                "status": "success",
                "results": {"results_total": 1, "completed": 1},
            },
            "frontend": {
                "status": "success",
                "metrics": {"answers_required": 1, "answers_received": 0},
            },
        }
        _prepare_runflow_files(run_dir, stages=stages)
        _prepare_validation_artifacts(run_dir, sid=sid, account_number=21)

        account_id = "idx-021"
        _prepare_review_manifest(run_dir, account_id)

        runflow.runflow_barriers_refresh(sid)

        payload = _load_runflow_payload(run_dir)
        umbrella = payload["umbrella_barriers"]

        assert umbrella["merge_ready"] is True
        assert umbrella["validation_ready"] is True
        assert umbrella["review_ready"] is False
        assert umbrella["all_ready"] is False
        assert payload["umbrella_ready"] is False

        _write_review_response(run_dir, account_id)
        updated_snapshot = _load_runflow_payload(run_dir)
        frontend_stage = updated_snapshot.setdefault("stages", {}).setdefault(
            "frontend", {}
        )
        metrics_payload = frontend_stage.setdefault("metrics", {})
        metrics_payload["answers_required"] = metrics_payload.get(
            "answers_required", 1
        )
        metrics_payload["answers_received"] = 1
        _write_json(run_dir / "runflow.json", updated_snapshot)
        runflow.runflow_barriers_refresh(sid)

        updated_payload = _load_runflow_payload(run_dir)
        updated_umbrella = updated_payload["umbrella_barriers"]

        assert updated_umbrella["merge_ready"] is True
        assert updated_umbrella["validation_ready"] is True
        assert updated_umbrella["review_ready"] is True
        assert updated_umbrella["all_ready"] is True
        assert updated_payload["umbrella_ready"] is True

        leftover_tmp = [path for path in run_dir.rglob("*.tmp")]
        assert not leftover_tmp
    finally:
        monkeypatch.delenv("RUNS_ROOT", raising=False)
        monkeypatch.delenv("UMBRELLA_BARRIERS_ENABLED", raising=False)
        monkeypatch.delenv("UMBRELLA_BARRIERS_LOG", raising=False)
        _reload_runflow()


def test_validation_zero_packs_marks_ready(tmp_path, monkeypatch):
    sid = "validation-empty"
    monkeypatch.setenv("RUNS_ROOT", str(tmp_path))
    monkeypatch.setenv("UMBRELLA_BARRIERS_ENABLED", "1")
    monkeypatch.setenv("UMBRELLA_BARRIERS_LOG", "0")
    _ensure_requests_stub(monkeypatch)
    _reload_runflow()

    try:
        index_dir = tmp_path / sid / "ai_packs" / "validation"
        index_dir.mkdir(parents=True, exist_ok=True)
        index_payload = {
            "schema_version": 2,
            "sid": sid,
            "root": ".",
            "packs_dir": "packs",
            "results_dir": "results",
            "packs": [],
        }
        _write_json(index_dir / "index.json", index_payload)

        import backend.runflow.decider as decider

        importlib.reload(decider)

        decider.refresh_validation_stage_from_index(sid, runs_root=tmp_path)
        decider.reconcile_umbrella_barriers(sid, runs_root=tmp_path)

        run_dir = tmp_path / sid
        payload = _load_runflow_payload(run_dir)
        validation_stage = payload["stages"]["validation"]
        assert validation_stage["status"] == "success"
        summary = validation_stage["summary"]
        assert summary["results_total"] == 0
        assert summary["completed"] == 0
        assert summary["empty_ok"] is True
        umbrella = payload["umbrella_barriers"]
        assert umbrella["validation_ready"] is True
    finally:
        monkeypatch.delenv("RUNS_ROOT", raising=False)
        monkeypatch.delenv("UMBRELLA_BARRIERS_ENABLED", raising=False)
        monkeypatch.delenv("UMBRELLA_BARRIERS_LOG", raising=False)
        _reload_runflow()


def test_validation_results_ready_for_built_status(tmp_path, monkeypatch):
    sid = "validation-built"
    monkeypatch.setenv("RUNS_ROOT", str(tmp_path))
    monkeypatch.setenv("UMBRELLA_BARRIERS_ENABLED", "1")
    monkeypatch.setenv("UMBRELLA_BARRIERS_LOG", "0")
    _ensure_requests_stub(monkeypatch)
    _reload_runflow()

    try:
        run_dir = tmp_path / sid
        validation_dir = run_dir / "ai_packs" / "validation"
        results_dir = validation_dir / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        result_filename = "idx-001.result.jsonl"
        (results_dir / result_filename).write_text("{}", encoding="utf-8")

        index_payload = {
            "schema_version": 2,
            "sid": sid,
            "root": ".",
            "packs_dir": "packs",
            "results_dir": "results",
            "packs": [
                {
                    "account_id": 1,
                    "pack": "packs/idx-001.json",
                    "result_json": f"results/{result_filename}",
                    "result_jsonl": f"results/{result_filename}",
                    "status": "built",
                    "lines": 0,
                }
            ],
        }
        _write_json(validation_dir / "index.json", index_payload)

        import backend.runflow.decider as decider

        importlib.reload(decider)

        decider.refresh_validation_stage_from_index(sid, runs_root=tmp_path)
        decider.reconcile_umbrella_barriers(sid, runs_root=tmp_path)

        payload = _load_runflow_payload(run_dir)
        validation_stage = payload["stages"]["validation"]
        assert validation_stage["status"] == "success"
        summary = validation_stage["summary"]
        assert summary["results_total"] == 1
        assert summary["completed"] == 1
        umbrella = payload["umbrella_barriers"]
        assert umbrella["validation_ready"] is True
    finally:
        monkeypatch.delenv("RUNS_ROOT", raising=False)
        monkeypatch.delenv("UMBRELLA_BARRIERS_ENABLED", raising=False)
        monkeypatch.delenv("UMBRELLA_BARRIERS_LOG", raising=False)
        _reload_runflow()


def test_frontend_zero_required_marks_ready(tmp_path, monkeypatch):
    sid = "frontend-empty"
    monkeypatch.setenv("RUNS_ROOT", str(tmp_path))
    monkeypatch.setenv("UMBRELLA_BARRIERS_ENABLED", "1")
    monkeypatch.setenv("UMBRELLA_BARRIERS_LOG", "0")
    _ensure_requests_stub(monkeypatch)
    _reload_runflow()

    try:
        import backend.runflow.decider as decider

        importlib.reload(decider)

        decider.refresh_frontend_stage_from_responses(sid, runs_root=tmp_path)
        decider.reconcile_umbrella_barriers(sid, runs_root=tmp_path)

        run_dir = tmp_path / sid
        payload = _load_runflow_payload(run_dir)
        frontend_stage = payload["stages"]["frontend"]
        assert frontend_stage["status"] == "success"
        summary = frontend_stage["summary"]
        assert summary["answers_required"] == 0
        assert summary["answers_received"] == 0
        assert summary["empty_ok"] is True
        umbrella = payload["umbrella_barriers"]
        assert umbrella["review_ready"] is True
    finally:
        monkeypatch.delenv("RUNS_ROOT", raising=False)
        monkeypatch.delenv("UMBRELLA_BARRIERS_ENABLED", raising=False)
        monkeypatch.delenv("UMBRELLA_BARRIERS_LOG", raising=False)
        _reload_runflow()


def test_merge_not_required_without_artifacts_marks_ready(tmp_path, monkeypatch):
    sid = "merge-optional"
    monkeypatch.setenv("RUNS_ROOT", str(tmp_path))
    monkeypatch.setenv("UMBRELLA_BARRIERS_ENABLED", "1")
    monkeypatch.setenv("UMBRELLA_BARRIERS_LOG", "0")
    monkeypatch.setenv("MERGE_REQUIRED", "0")
    _ensure_requests_stub(monkeypatch)
    _reload_runflow()

    try:
        import backend.runflow.decider as decider

        importlib.reload(decider)

        statuses = decider.reconcile_umbrella_barriers(sid, runs_root=tmp_path)
        assert statuses["merge_ready"] is True

        payload = _load_runflow_payload(tmp_path / sid)
        umbrella = payload["umbrella_barriers"]
        assert umbrella["merge_ready"] is True
    finally:
        monkeypatch.delenv("RUNS_ROOT", raising=False)
        monkeypatch.delenv("UMBRELLA_BARRIERS_ENABLED", raising=False)
        monkeypatch.delenv("UMBRELLA_BARRIERS_LOG", raising=False)
        monkeypatch.delenv("MERGE_REQUIRED", raising=False)
        _reload_runflow()


def test_document_barrier_flag_emitted_when_enabled(tmp_path, monkeypatch):
    sid = "documents-enabled"
    monkeypatch.setenv("RUNS_ROOT", str(tmp_path))
    monkeypatch.setenv("UMBRELLA_BARRIERS_ENABLED", "1")
    monkeypatch.setenv("UMBRELLA_BARRIERS_LOG", "0")
    monkeypatch.setenv("DOCUMENT_VERIFIER_ENABLED", "1")
    _ensure_requests_stub(monkeypatch)
    _reload_runflow()

    try:
        run_dir = tmp_path / sid
        stages: dict[str, dict] = {}
        _prepare_runflow_files(run_dir, stages=stages)

        runflow.runflow_barriers_refresh(sid)

        payload = _load_runflow_payload(run_dir)
        umbrella = payload["umbrella_barriers"]

        assert umbrella["merge_ready"] is False
        assert umbrella["validation_ready"] is False
        assert umbrella["review_ready"] is False
        assert umbrella["all_ready"] is False
        assert umbrella["document_ready"] is False
        assert payload["umbrella_ready"] is False
    finally:
        monkeypatch.delenv("RUNS_ROOT", raising=False)
        monkeypatch.delenv("UMBRELLA_BARRIERS_ENABLED", raising=False)
        monkeypatch.delenv("UMBRELLA_BARRIERS_LOG", raising=False)
        monkeypatch.delenv("DOCUMENT_VERIFIER_ENABLED", raising=False)
        _reload_runflow()
