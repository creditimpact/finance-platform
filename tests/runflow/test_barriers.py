from __future__ import annotations

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
    default_payload = {"sid": run_dir.name, "note": "persist"}
    payload = runflow_payload or default_payload
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
        stages = {"merge": {"status": "success"}}
        _prepare_runflow_files(run_dir, stages=stages)

        before = {entry.name for entry in run_dir.iterdir()}
        runflow._update_umbrella_barriers(sid)
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
            counts={},
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
        assert not (run_dir / "runflow_steps.json").exists()
    finally:
        monkeypatch.delenv("RUNS_ROOT", raising=False)
        monkeypatch.delenv("UMBRELLA_BARRIERS_ENABLED", raising=False)
        monkeypatch.delenv("UMBRELLA_BARRIERS_LOG", raising=False)
        monkeypatch.delenv("RUNFLOW_VERBOSE", raising=False)
        monkeypatch.delenv("RUNFLOW_EVENTS", raising=False)
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
        stages = {"validation": {"status": "success"}}
        _prepare_runflow_files(run_dir, stages=stages)
        _prepare_validation_artifacts(run_dir, sid=sid, account_number=42)

        runflow._update_umbrella_barriers(sid)
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
            "merge": {"status": "success"},
            "validation": {"status": "success"},
        }
        _prepare_runflow_files(run_dir, stages=stages)
        _prepare_validation_artifacts(run_dir, sid=sid, account_number=7)

        account_id = "idx-007"
        _prepare_review_manifest(run_dir, account_id)
        _write_review_response(run_dir, account_id)

        runflow._update_umbrella_barriers(sid)

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
            "merge": {"status": "success"},
            "validation": {"status": "success"},
        }
        _prepare_runflow_files(run_dir, stages=stages)
        _prepare_validation_artifacts(run_dir, sid=sid, account_number=21)

        account_id = "idx-021"
        _prepare_review_manifest(run_dir, account_id)

        runflow._update_umbrella_barriers(sid)

        payload = _load_runflow_payload(run_dir)
        umbrella = payload["umbrella_barriers"]

        assert umbrella["merge_ready"] is True
        assert umbrella["validation_ready"] is True
        assert umbrella["review_ready"] is False
        assert umbrella["all_ready"] is False
        assert payload["umbrella_ready"] is False

        _write_review_response(run_dir, account_id)
        runflow._update_umbrella_barriers(sid)

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

        runflow._update_umbrella_barriers(sid)

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
