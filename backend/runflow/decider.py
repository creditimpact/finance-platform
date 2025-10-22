"""Simple runflow state machine for deciding the next pipeline action."""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Literal, Mapping, Optional

from backend.core.io.json_io import _atomic_write_json
from backend.core.runflow import (
    runflow_decide_step,
    runflow_end_stage,
    runflow_refresh_umbrella_barriers,
)
from backend.runflow.counters import (
    frontend_answers_counters as _frontend_answers_counters,
    stage_counts as _stage_counts_from_disk,
)
from backend.validation.index_schema import load_validation_index

StageStatus = Literal["success", "error", "built", "published"]
RunState = Literal[
    "INIT",
    "VALIDATING",
    "AWAITING_CUSTOMER_INPUT",
    "COMPLETE_NO_ACTION",
    "ERROR",
]
StageName = Literal["merge", "validation", "frontend"]


log = logging.getLogger(__name__)

_RUNFLOW_FILENAME = "runflow.json"


def _env_enabled(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"", "0", "false", "no", "off"}


def _umbrella_barriers_enabled() -> bool:
    return _env_enabled("UMBRELLA_BARRIERS_ENABLED", True)


def _strict_validation_enabled() -> bool:
    return _env_enabled("UMBRELLA_BARRIERS_STRICT_VALIDATION", True)


def _review_explanation_required() -> bool:
    return _env_enabled("UMBRELLA_BARRIERS_REVIEW_REQUIRE_EXPLANATION", True)


def _review_attachment_required() -> bool:
    return _env_enabled("UMBRELLA_BARRIERS_REVIEW_REQUIRE_FILE", False)


def _validation_autosend_enabled() -> bool:
    return _env_enabled("VALIDATION_AUTOSEND", True)


def _barrier_event_logging_enabled() -> bool:
    return _env_enabled("UMBRELLA_BARRIERS_LOG", True)


def _document_verifier_enabled() -> bool:
    return _env_enabled("DOCUMENT_VERIFIER_ENABLED", False)


def _default_runs_root() -> Path:
    root_env = os.getenv("RUNS_ROOT")
    return Path(root_env) if root_env else Path("runs")


def _now_iso() -> str:
    """Return an ISO-8601 timestamp in UTC with second precision."""

    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _resolve_runs_root(runs_root: Optional[str | Path]) -> Path:
    if runs_root is None:
        return _default_runs_root()
    return Path(runs_root)


def _runflow_path(sid: str, runs_root: Optional[str | Path]) -> Path:
    base = _resolve_runs_root(runs_root) / sid
    base.mkdir(parents=True, exist_ok=True)
    return base / _RUNFLOW_FILENAME


def _coerce_int(value: Any) -> Optional[int]:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _default_umbrella_barriers() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "merge_ready": False,
        "validation_ready": False,
        "review_ready": False,
        "all_ready": False,
        "checked_at": None,
    }

    if _document_verifier_enabled():
        payload["document_ready"] = False

    return payload


def _normalise_umbrella_barriers(payload: Any) -> dict[str, Any]:
    result = _default_umbrella_barriers()
    if isinstance(payload, Mapping):
        for key in ("merge_ready", "validation_ready", "review_ready", "all_ready"):
            value = payload.get(key)
            if isinstance(value, bool):
                result[key] = value
        checked_at = payload.get("checked_at")
        if isinstance(checked_at, str):
            result["checked_at"] = checked_at

        document_ready = payload.get("document_ready")
        if isinstance(document_ready, bool):
            result["document_ready"] = document_ready
    return result


def _load_runflow(path: Path, sid: str) -> dict[str, Any]:
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        data: dict[str, Any] = {
            "sid": sid,
            "run_state": "INIT",
            "stages": {},
            "updated_at": _now_iso(),
            "umbrella_barriers": _default_umbrella_barriers(),
        }
        return data
    except OSError:
        log.warning("RUNFLOW_READ_FAILED sid=%s path=%s", sid, path, exc_info=True)
        return {
            "sid": sid,
            "run_state": "INIT",
            "stages": {},
            "updated_at": _now_iso(),
            "umbrella_barriers": _default_umbrella_barriers(),
        }

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        log.warning("RUNFLOW_PARSE_FAILED sid=%s path=%s", sid, path, exc_info=True)
        payload = {}

    if not isinstance(payload, Mapping):
        payload = {}

    stages = payload.get("stages")
    if not isinstance(stages, Mapping):
        stages = {}

    run_state = payload.get("run_state")
    if not isinstance(run_state, str) or run_state not in {
        "INIT",
        "VALIDATING",
        "AWAITING_CUSTOMER_INPUT",
        "COMPLETE_NO_ACTION",
        "ERROR",
    }:
        run_state = "INIT"

    umbrella_barriers = _normalise_umbrella_barriers(payload.get("umbrella_barriers"))

    return {
        "sid": sid,
        "run_state": run_state,
        "stages": dict(stages),
        "updated_at": str(payload.get("updated_at") or _now_iso()),
        "umbrella_barriers": umbrella_barriers,
    }


def record_stage(
    sid: str,
    stage: StageName,
    *,
    status: StageStatus,
    counts: Dict[str, int],
    empty_ok: bool,
    notes: Optional[str] = None,
    metrics: Optional[Mapping[str, Any]] = None,
    results: Optional[Mapping[str, Any]] = None,
    runs_root: Optional[str | Path] = None,
) -> dict[str, Any]:
    """Persist ``stage`` information under ``runs/<sid>/runflow.json``."""

    path = _runflow_path(sid, runs_root)
    base_dir = path.parent
    data = _load_runflow(path, sid)

    stage_payload: dict[str, Any] = {
        "status": status,
        "empty_ok": bool(empty_ok),
        "last_at": _now_iso(),
    }

    normalized_counts: dict[str, int] = {}
    for key, value in (counts or {}).items():
        coerced = _coerce_int(value)
        if coerced is not None:
            normalized_counts[str(key)] = coerced

    def _normalize_mapping(payload: Optional[Mapping[str, Any]]) -> dict[str, int]:
        normalized: dict[str, int] = {}
        if not isinstance(payload, Mapping):
            return normalized
        for key, value in payload.items():
            coerced_value = _coerce_int(value)
            if coerced_value is not None:
                normalized[str(key)] = coerced_value
        return normalized

    normalized_metrics = _normalize_mapping(metrics)
    normalized_results = _normalize_mapping(results)

    if stage == "frontend":
        answers_metrics = _frontend_answers_counters(
            base_dir, attachments_required=_review_attachment_required()
        )
        answers_required = _coerce_int(answers_metrics.get("answers_required"))
        answers_received = _coerce_int(answers_metrics.get("answers_received"))
        if answers_required is not None:
            normalized_metrics["answers_required"] = answers_required
        if answers_received is not None:
            normalized_metrics["answers_received"] = answers_received

    disk_counts = _stage_counts_from_disk(stage, base_dir)
    for key, value in disk_counts.items():
        coerced_disk = _coerce_int(value)
        if coerced_disk is None:
            continue
        existing = normalized_counts.get(str(key))
        if isinstance(existing, int) and existing > 0 and coerced_disk == 0:
            continue
        normalized_counts[str(key)] = coerced_disk

    for key, value in normalized_counts.items():
        stage_payload[key] = value

    if normalized_metrics:
        stage_payload["metrics"] = dict(normalized_metrics)

    if normalized_results:
        stage_payload["results"] = dict(normalized_results)

    if notes:
        stage_payload["notes"] = str(notes)

    stages = data.setdefault("stages", {})
    if not isinstance(stages, dict):
        stages = {}
        data["stages"] = stages
    stages[stage] = stage_payload

    if "umbrella_barriers" not in data or not isinstance(data["umbrella_barriers"], dict):
        data["umbrella_barriers"] = _default_umbrella_barriers()

    # ``runflow_end_stage`` may update the on-disk payload (via
    # ``_update_umbrella_barriers``) before we persist ``data``. Merge the
    # latest umbrella readiness flags so stage writes do not clobber barrier
    # evaluations that just ran.
    try:
        existing_raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        existing_payload: Mapping[str, Any] | None = None
    except OSError:
        existing_payload = None
    else:
        try:
            parsed = json.loads(existing_raw)
        except json.JSONDecodeError:
            existing_payload = None
        else:
            existing_payload = parsed if isinstance(parsed, Mapping) else None

    if existing_payload:
        existing_barriers = existing_payload.get("umbrella_barriers")
        if isinstance(existing_barriers, Mapping):
            merged_barriers = dict(existing_barriers)
            current_barriers = data.get("umbrella_barriers")
            if isinstance(current_barriers, Mapping):
                merged_barriers.update(current_barriers)
            data["umbrella_barriers"] = merged_barriers

    if status == "error":
        data["run_state"] = "ERROR"
    elif stage == "validation" and data.get("run_state") == "INIT":
        data["run_state"] = "VALIDATING"

    data["updated_at"] = _now_iso()

    log.info(
        "RUNFLOW_RECORD sid=%s stage=%s status=%s counts=%s empty_ok=%s",
        sid,
        stage,
        status,
        dict(normalized_counts),
        empty_ok,
    )

    summary_payload: dict[str, Any] = {str(key): value for key, value in normalized_counts.items()}
    if normalized_metrics:
        summary_payload["metrics"] = dict(normalized_metrics)
    if normalized_results:
        summary_payload["results"] = dict(normalized_results)
    summary_payload["empty_ok"] = bool(empty_ok)
    if notes:
        summary_payload["notes"] = str(notes)

    stage_status_override: Optional[str] = None
    if stage == "frontend" and status != "error":
        packs_value: Optional[int] = None
        for key in ("packs_count", "packs"):
            packs_value = normalized_counts.get(key)
            if packs_value is not None:
                break
        if packs_value == 0:
            stage_status_override = "empty"

    runflow_end_stage(
        sid,
        stage,
        status=status,
        summary=summary_payload if summary_payload else None,
        stage_status=stage_status_override,
        empty_ok=empty_ok,
    )

    _atomic_write_json(path, data)
    runflow_refresh_umbrella_barriers(sid)
    return data


def finalize_merge_stage(
    sid: str,
    *,
    runs_root: Optional[str | Path] = None,
    notes: Optional[str] = None,
) -> dict[str, Any]:
    """Mark the merge stage as complete using authoritative on-disk data."""

    base_root = _resolve_runs_root(runs_root)
    base_dir = base_root / sid
    merge_dir = base_dir / "ai_packs" / "merge"

    index_path = merge_dir / "pairs_index.json"
    index_payload = _load_json_mapping(index_path)
    if not isinstance(index_payload, Mapping):
        index_payload = {}

    totals_payload = index_payload.get("totals")
    totals = dict(totals_payload) if isinstance(totals_payload, Mapping) else {}

    def _maybe_int(value: Any) -> Optional[int]:
        coerced = _coerce_int(value)
        return coerced if coerced is not None else None

    metrics: dict[str, int] = {}
    for key in (
        "scored_pairs",
        "matches_strong",
        "matches_weak",
        "conflicts",
        "skipped",
        "packs_built",
        "created_packs",
        "topn_limit",
        "normalized_accounts",
    ):
        coerced = _maybe_int(totals.get(key))
        if coerced is not None:
            metrics[key] = coerced

    if "scored_pairs" not in metrics:
        fallback_scored = _maybe_int(index_payload.get("scored_pairs"))
        if fallback_scored is not None:
            metrics["scored_pairs"] = fallback_scored
    if "created_packs" not in metrics:
        fallback_created = _maybe_int(index_payload.get("created_packs"))
        if fallback_created is not None:
            metrics["created_packs"] = fallback_created

    pairs_payload = index_payload.get("pairs")
    if isinstance(pairs_payload, list):
        metrics["pairs_index_entries"] = len(pairs_payload)
        if "created_packs" not in metrics:
            metrics["created_packs"] = len(pairs_payload)

    results_dir = merge_dir / "results"
    try:
        result_files_total = sum(
            1 for path in results_dir.rglob("*.result.json") if path.is_file()
        )
    except OSError:
        result_files_total = 0

    metrics["result_files"] = result_files_total

    packs_dir = merge_dir / "packs"
    try:
        pack_files_total = sum(
            1 for path in packs_dir.glob("pair_*.jsonl") if path.is_file()
        )
    except OSError:
        pack_files_total = 0

    metrics["pack_files"] = pack_files_total

    scored_pairs_value = metrics.get("scored_pairs")
    if scored_pairs_value is None:
        scored_pairs_value = 0
        metrics["scored_pairs"] = 0

    existing_created = metrics.get("created_packs")
    candidate_created = [result_files_total, pack_files_total]
    if isinstance(existing_created, int):
        candidate_created.append(existing_created)
    created_packs_value = max(candidate_created)
    metrics["created_packs"] = created_packs_value

    counts: dict[str, int] = {
        "pairs_scored": scored_pairs_value,
        "packs_created": created_packs_value,
        "result_files": result_files_total,
    }

    empty_ok = created_packs_value == 0 or scored_pairs_value == 0

    results_payload = {"result_files": result_files_total}

    record_stage(
        sid,
        "merge",
        status="success",
        counts=counts,
        empty_ok=empty_ok,
        metrics=metrics,
        results=results_payload,
        runs_root=base_root,
        notes=notes,
    )

    return {
        "counts": counts,
        "metrics": metrics,
        "results": results_payload,
        "empty_ok": empty_ok,
    }


def _load_json_mapping(path: Path) -> Mapping[str, Any] | None:
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError:
        log.warning("RUNFLOW_JSON_READ_FAILED path=%s", path, exc_info=True)
        return None

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        log.warning("RUNFLOW_JSON_PARSE_FAILED path=%s", path, exc_info=True)
        return None

    if isinstance(payload, Mapping):
        return payload

    return None


def _get_stage_info(
    stages: Mapping[str, Any] | None, stage: str
) -> Mapping[str, Any] | None:
    if not isinstance(stages, Mapping):
        return None
    candidate = stages.get(stage)
    if isinstance(candidate, Mapping):
        return candidate
    return None


def _stage_has_counters(stage_info: Mapping[str, Any] | None) -> bool:
    if not isinstance(stage_info, Mapping):
        return False

    skip_keys = {
        "status",
        "empty_ok",
        "last_at",
        "notes",
        "metrics",
        "results",
        "error",
        "summary",
    }

    for key, value in stage_info.items():
        if key in skip_keys:
            continue
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            return True
    return False


def _stage_metrics_value(stage_info: Mapping[str, Any] | None, key: str) -> Optional[int]:
    if not isinstance(stage_info, Mapping):
        return None
    metrics = stage_info.get("metrics")
    if not isinstance(metrics, Mapping):
        return None
    return _coerce_int(metrics.get(key))


def _stage_status(steps: Mapping[str, Any] | None, stage: str) -> str:
    if not isinstance(steps, Mapping):
        return ""

    stage_info = steps.get(stage)
    if not isinstance(stage_info, Mapping):
        return ""

    status = stage_info.get("status")
    if isinstance(status, str):
        return status.strip().lower()

    return ""


def _resolve_validation_index_path(run_dir: Path) -> Path:
    override = os.getenv("VALIDATION_INDEX_PATH")
    if override:
        candidate = Path(override)
        if not candidate.is_absolute():
            candidate = (run_dir / override).resolve()
        else:
            candidate = candidate.resolve()
        return candidate

    return (run_dir / "ai_packs" / "validation" / "index.json").resolve()


def _resolve_validation_results_dir(run_dir: Path) -> Optional[Path]:
    override = os.getenv("VALIDATION_RESULTS_DIR")
    if not override:
        return None

    candidate = Path(override)
    if not candidate.is_absolute():
        candidate = (run_dir / override).resolve()
    else:
        candidate = candidate.resolve()
    return candidate


def _validation_record_result_paths(
    index: "ValidationIndex", record: Any, *, results_override: Optional[Path] = None
) -> list[Path]:  # pragma: no cover - exercised via higher level tests
    paths: list[Path] = []
    seen: set[str] = set()

    def _append(path: Path) -> None:
        key = str(path)
        if key not in seen:
            seen.add(key)
            paths.append(path)

    result_json_value = getattr(record, "result_json", None)
    if isinstance(result_json_value, str) and result_json_value.strip():
        try:
            _append(index.resolve_result_json_path(record))
        except Exception:
            return []

    result_jsonl_value = getattr(record, "result_jsonl", None)
    if isinstance(result_jsonl_value, str) and result_jsonl_value.strip():
        try:
            _append(index.resolve_result_jsonl_path(record))
        except Exception:
            return []

    if results_override:
        base_dir: Optional[Path]
        try:
            base_dir = index.results_dir_path
        except Exception:  # pragma: no cover - defensive
            base_dir = None

        for candidate in list(paths):
            relative: Optional[Path] = None
            if base_dir is not None:
                try:
                    relative = candidate.resolve().relative_to(base_dir)
                except ValueError:
                    relative = None
            if relative is None:
                relative = Path(candidate.name)

            override_candidate = (results_override / relative).resolve()
            _append(override_candidate)

    return paths


def _validation_record_has_results(
    index: "ValidationIndex", record: Any, *, results_override: Optional[Path] = None
) -> bool:
    paths = _validation_record_result_paths(
        index, record, results_override=results_override
    )
    if not paths:
        return False

    for candidate in paths:
        try:
            if candidate.is_file():
                return True
        except OSError:
            continue
    return False


def _validation_results_progress(run_dir: Path) -> tuple[int, int, int, bool]:
    """Return (total, completed, failed, ready) for validation AI results."""

    index_path = _resolve_validation_index_path(run_dir)
    results_override = _resolve_validation_results_dir(run_dir)
    try:
        index = load_validation_index(index_path)
    except FileNotFoundError:
        return (0, 0, 0, False)
    except Exception:  # pragma: no cover - defensive
        log.warning("RUNFLOW_VALIDATION_INDEX_LOAD_FAILED path=%s", index_path, exc_info=True)
        return (0, 0, 0, False)

    total = 0
    completed = 0
    failed = 0
    ready = True

    for record in getattr(index, "packs", ()):  # type: ignore[attr-defined]
        total += 1
        status = getattr(record, "status", "")
        normalized_status = status.strip().lower() if isinstance(status, str) else ""

        if normalized_status == "completed":
            if not _validation_record_has_results(
                index, record, results_override=results_override
            ):
                log.warning(
                    "RUNFLOW_VALIDATION_RESULT_MISSING account_id=%s",
                    getattr(record, "account_id", "unknown"),
                )
                ready = False
                continue

            completed += 1
        elif normalized_status == "failed":
            failed += 1
            ready = False
        else:
            ready = False

    if total == 0:
        return (0, 0, 0, False)

    ready = ready and completed == total
    return (total, completed, failed, ready)


def _validation_stage_ready(
    run_dir: Path, validation_stage: Mapping[str, Any] | None
) -> bool:
    total, completed, failed, ready = _validation_results_progress(run_dir)

    if total > 0:
        return ready and completed == total

    if not isinstance(validation_stage, Mapping):
        return False

    findings_count = _coerce_int(validation_stage.get("findings_count"))
    if findings_count is None or findings_count != 0:
        return False

    packs_total = _stage_metrics_value(validation_stage, "packs_total")
    if packs_total is None:
        return False

    if not _validation_autosend_enabled():
        return packs_total == 0

    return packs_total == 0


_IDX_ACCOUNT_PATTERN = re.compile(r"idx-(\d+)")


def _response_filename_for_account(account_id: str) -> str:
    trimmed = (account_id or "").strip()
    match = _IDX_ACCOUNT_PATTERN.fullmatch(trimmed)
    number: int | None = None
    if match:
        number = int(match.group(1))
    else:
        try:
            number = int(trimmed)
        except ValueError:
            number = None

    if number is not None:
        return f"idx-{number:03d}.result.json"

    sanitized = re.sub(r"[^A-Za-z0-9_.-]", "_", trimmed) or "account"
    return f"{sanitized}.result.json"


def _frontend_responses_progress(run_dir: Path) -> tuple[int, int, bool]:
    attachments_required = _review_attachment_required()
    counters = _frontend_answers_counters(
        run_dir, attachments_required=attachments_required
    )

    required = _coerce_int(counters.get("answers_required")) or 0
    answered = _coerce_int(counters.get("answers_received")) or 0
    ready = answered == required
    return (required, answered, ready)


def _review_responses_ready(run_dir: Path) -> bool:
    _, _, ready = _frontend_responses_progress(run_dir)
    return ready


def _compute_umbrella_barriers(run_dir: Path) -> dict[str, bool]:
    runflow_path = run_dir / "runflow.json"
    runflow_payload = _load_json_mapping(runflow_path)
    runflow_stages = (
        runflow_payload.get("stages")
        if isinstance(runflow_payload, Mapping)
        else None
    )

    steps_path = run_dir / "runflow_steps.json"
    steps_payload = _load_json_mapping(steps_path)
    steps_stages = (
        steps_payload.get("stages") if isinstance(steps_payload, Mapping) else None
    )

    def _combined_stage_status(stage_name: str) -> str:
        status = _stage_status(runflow_stages, stage_name)
        if status:
            return status
        return _stage_status(steps_stages, stage_name)

    merge_stage_info = _get_stage_info(runflow_stages, "merge")
    if merge_stage_info is None:
        merge_stage_info = _get_stage_info(steps_stages, "merge")
    merge_status = (_combined_stage_status("merge") or "").strip().lower()
    merge_empty_ok = bool(merge_stage_info.get("empty_ok")) if isinstance(
        merge_stage_info, Mapping
    ) else False

    counts_from_disk = _stage_counts_from_disk("merge", run_dir)
    has_disk_counts = any(
        isinstance(value, (int, float)) and not isinstance(value, bool)
        for value in counts_from_disk.values()
    )
    has_recorded_counts = _stage_has_counters(merge_stage_info) or has_disk_counts

    if merge_status == "error":
        merge_ready = False
    elif merge_empty_ok:
        merge_ready = True
    else:
        merge_ready = merge_status not in {"", "running"} and has_recorded_counts

    validation_stage_info = _get_stage_info(runflow_stages, "validation")
    if validation_stage_info is None:
        validation_stage_info = _get_stage_info(steps_stages, "validation")
    validation_status = (_combined_stage_status("validation") or "").strip().lower()
    validation_ready = _validation_stage_ready(run_dir, validation_stage_info)
    if validation_status == "error":
        validation_ready = False

    review_stage_info = _get_stage_info(runflow_stages, "frontend")
    if review_stage_info is None:
        review_stage_info = _get_stage_info(steps_stages, "frontend")
    review_status = (_combined_stage_status("frontend") or "").strip().lower()
    _, _, review_ready = _frontend_responses_progress(run_dir)
    if review_status == "error":
        review_ready = False
    if isinstance(review_stage_info, Mapping) and bool(review_stage_info.get("empty_ok")):
        review_ready = True

    all_ready = merge_ready and validation_ready and review_ready

    readiness: dict[str, bool] = {
        "merge_ready": merge_ready,
        "validation_ready": validation_ready,
        "review_ready": review_ready,
        "all_ready": all_ready,
    }

    if _document_verifier_enabled():
        readiness["document_ready"] = False

    return readiness


def refresh_validation_stage_from_index(
    sid: str, runs_root: Optional[str | Path] = None
) -> None:
    """Update the validation stage entry when AI results are complete."""

    path = _runflow_path(sid, runs_root)
    run_dir = path.parent
    total, completed, failed, ready = _validation_results_progress(run_dir)
    if not ready:
        return

    data = _load_runflow(path, sid)
    stages = data.setdefault("stages", {})
    if not isinstance(stages, dict):
        stages = {}
        data["stages"] = stages

    existing = stages.get("validation")
    stage_payload = dict(existing) if isinstance(existing, Mapping) else {}

    stage_payload["status"] = "success"
    stage_payload["last_at"] = _now_iso()
    if "empty_ok" not in stage_payload:
        stage_payload["empty_ok"] = bool(total == 0)

    results_payload = {
        "results_total": total,
        "completed": completed,
        "failed": failed,
    }
    stage_payload["results"] = results_payload

    stages["validation"] = stage_payload
    data["updated_at"] = _now_iso()

    _atomic_write_json(path, data)
    runflow_refresh_umbrella_barriers(sid)


def refresh_frontend_stage_from_responses(
    sid: str, runs_root: Optional[str | Path] = None
) -> None:
    """Update the frontend stage entry when customer responses are complete."""

    path = _runflow_path(sid, runs_root)
    run_dir = path.parent
    required, answered, ready = _frontend_responses_progress(run_dir)
    if not ready:
        return

    data = _load_runflow(path, sid)
    stages = data.setdefault("stages", {})
    if not isinstance(stages, dict):
        stages = {}
        data["stages"] = stages

    existing = stages.get("frontend")
    stage_payload = dict(existing) if isinstance(existing, Mapping) else {}

    stage_payload["status"] = "success"
    stage_payload["last_at"] = _now_iso()
    if "empty_ok" not in stage_payload:
        stage_payload["empty_ok"] = bool(required == 0)

    metrics_payload = stage_payload.get("metrics")
    if isinstance(metrics_payload, Mapping):
        metrics_data = dict(metrics_payload)
    else:
        metrics_data = {}
    metrics_data["answers_received"] = answered
    metrics_data["answers_required"] = required
    stage_payload["metrics"] = metrics_data
    stage_payload.pop("answers", None)

    stages["frontend"] = stage_payload
    data["updated_at"] = _now_iso()

    _atomic_write_json(path, data)
    runflow_refresh_umbrella_barriers(sid)


def reconcile_umbrella_barriers(
    sid: str, runs_root: Optional[str | Path] = None
) -> dict[str, bool]:
    """Recompute umbrella readiness for ``sid`` and persist the booleans."""

    runflow_path = _runflow_path(sid, runs_root)
    run_dir = runflow_path.parent
    statuses = _compute_umbrella_barriers(run_dir)

    data = _load_runflow(runflow_path, sid)
    existing = data.get("umbrella_barriers")
    if isinstance(existing, Mapping):
        umbrella = dict(existing)
    else:
        umbrella = {}

    timestamp = _now_iso()
    for key, value in statuses.items():
        if isinstance(key, str):
            umbrella[key] = bool(value)
    umbrella["checked_at"] = timestamp

    data["umbrella_barriers"] = umbrella
    data["umbrella_ready"] = bool(statuses.get("all_ready"))
    data["updated_at"] = timestamp

    _atomic_write_json(runflow_path, data)

    if _barrier_event_logging_enabled():
        events_path = run_dir / "runflow_events.jsonl"
        event_payload = {"ts": timestamp, "event": "barriers_reconciled"}
        for key, value in statuses.items():
            if isinstance(key, str):
                event_payload[key] = bool(value)
        try:
            events_path.parent.mkdir(parents=True, exist_ok=True)
            with events_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event_payload, ensure_ascii=False))
                handle.write("\n")
        except OSError:
            log.warning(
                "RUNFLOW_BARRIERS_EVENT_WRITE_FAILED sid=%s path=%s",
                sid,
                events_path,
                exc_info=True,
            )

    return statuses


def evaluate_global_barriers(run_path: str) -> dict[str, bool]:
    """Inspect run artifacts and report readiness for umbrella arguments."""

    if not _umbrella_barriers_enabled():
        return {
            "merge_ready": False,
            "validation_ready": False,
            "review_ready": False,
            "all_ready": False,
        }

    run_dir = Path(run_path)
    return _compute_umbrella_barriers(run_dir)


def _parse_timestamp(value: Any) -> Optional[datetime]:
    if not isinstance(value, str):
        return None
    try:
        if value.endswith("Z"):
            value = value[:-1] + "+00:00"
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _latest_stage_name(stages: Mapping[str, Mapping[str, Any]]) -> Optional[str]:
    stage_order = {"merge": 0, "validation": 1, "frontend": 2}
    best_name: Optional[str] = None
    best_ts: Optional[datetime] = None

    for name, info in stages.items():
        if not isinstance(info, Mapping):
            continue

        ts = (
            _parse_timestamp(info.get("last_at"))
            or _parse_timestamp(info.get("ended_at"))
            or _parse_timestamp(info.get("started_at"))
        )

        if ts is None:
            continue

        if best_ts is None or ts > best_ts or (
            ts == best_ts
            and stage_order.get(str(name), -1) >= stage_order.get(best_name or "", -1)
        ):
            best_name = str(name)
            best_ts = ts

    if best_name is not None:
        return best_name

    for candidate in ("frontend", "validation", "merge"):
        if candidate in stages:
            return candidate

    for name in stages:
        if isinstance(name, str):
            return name

    return None


def _decision_next_label(next_action: str) -> str:
    mapping = {
        "run_validation": "continue",
        "gen_frontend_packs": "run_frontend",
        "await_input": "await_input",
        "complete_no_action": "done",
        "stop_error": "done",
    }
    return mapping.get(next_action, "continue")


def decide_next(sid: str, runs_root: Optional[str | Path] = None) -> dict[str, str]:
    """Return the next pipeline action for ``sid`` based on recorded stages."""

    path = _runflow_path(sid, runs_root)
    data = _load_runflow(path, sid)
    stages: Mapping[str, Mapping[str, Any]] = data.get("stages", {})  # type: ignore[assignment]

    next_action = "run_validation"
    reason = "validation_pending"
    new_state: RunState = "VALIDATING"

    def _set(next_value: str, reason_value: str, state_value: RunState) -> None:
        nonlocal next_action, reason, new_state
        next_action = next_value
        reason = reason_value
        new_state = state_value

    for stage_name, stage_info in stages.items():
        if not isinstance(stage_info, Mapping):
            continue
        status = str(stage_info.get("status") or "")
        if status == "error":
            _set("stop_error", f"{stage_name}_error", "ERROR")
            break
    else:
        merge_stage = stages.get("merge")
        accounts_count: Optional[int] = None
        if isinstance(merge_stage, Mapping):
            for key in ("count", "accounts_count", "total_accounts"):
                accounts_count = _coerce_int(merge_stage.get(key))
                if accounts_count is not None:
                    break

        if accounts_count == 0:
            _set("complete_no_action", "no_accounts", "COMPLETE_NO_ACTION")
        else:
            validation_stage = stages.get("validation")
            if not isinstance(validation_stage, Mapping):
                _set("run_validation", "validation_pending", "VALIDATING")
            else:
                validation_status = str(validation_stage.get("status") or "")
                normalized_validation_status = validation_status.strip().lower()
                findings_count = _coerce_int(validation_stage.get("findings_count"))
                if findings_count is None:
                    findings_count = 0

                if normalized_validation_status == "error":
                    _set("stop_error", "validation_error", "ERROR")
                elif normalized_validation_status not in {"success", "built"}:
                    _set("run_validation", "validation_pending", "VALIDATING")
                elif findings_count <= 0:
                    _set("complete_no_action", "validation_no_findings", "COMPLETE_NO_ACTION")
                else:
                    frontend_stage = stages.get("frontend")
                    if not isinstance(frontend_stage, Mapping):
                        _set("gen_frontend_packs", "validation_has_findings", "VALIDATING")
                    else:
                        frontend_status = str(frontend_stage.get("status") or "")
                        normalized_frontend_status = frontend_status.strip().lower()
                        if normalized_frontend_status == "error":
                            _set("stop_error", "frontend_error", "ERROR")
                        elif normalized_frontend_status in {"published", "success"}:
                            packs_count = _coerce_int(frontend_stage.get("packs_count")) or 0
                            if packs_count <= 0:
                                _set(
                                    "complete_no_action",
                                    "frontend_no_packs",
                                    "COMPLETE_NO_ACTION",
                                )
                            else:
                                reason_label = (
                                    "frontend_completed"
                                    if normalized_frontend_status == "success"
                                    else "frontend_published"
                                )
                                _set(
                                    "await_input",
                                    reason_label,
                                    "AWAITING_CUSTOMER_INPUT",
                                )
                        else:
                            _set("gen_frontend_packs", "validation_has_findings", "VALIDATING")

    if data.get("run_state") != new_state:
        data["run_state"] = new_state
        data["updated_at"] = _now_iso()
        _atomic_write_json(path, data)
        try:
            reconcile_umbrella_barriers(sid, runs_root=runs_root)
        except Exception:  # pragma: no cover - defensive logging
            log.warning(
                "RUNFLOW_RECONCILE_AT_STATE_CHANGE_FAILED sid=%s state=%s",
                sid,
                new_state,
                exc_info=True,
            )

    log.info(
        "RUNFLOW_DECIDE sid=%s next=%s reason=%s state=%s",
        sid,
        next_action,
        reason,
        new_state,
    )

    stage_for_step = _latest_stage_name(stages)
    if stage_for_step and isinstance(stages.get(stage_for_step), Mapping):
        try:
            runflow_decide_step(
                sid,
                stage_for_step,
                next_action=_decision_next_label(next_action),
                reason=reason,
            )
        except Exception:  # pragma: no cover - defensive logging
            log.warning(
                "RUNFLOW_DECIDE_STEP_FAILED sid=%s stage=%s next=%s reason=%s",
                sid,
                stage_for_step,
                next_action,
                reason,
                exc_info=True,
            )

    return {"next": next_action, "reason": reason}


__all__ = [
    "record_stage",
    "decide_next",
    "StageStatus",
    "RunState",
    "refresh_validation_stage_from_index",
    "refresh_frontend_stage_from_responses",
    "reconcile_umbrella_barriers",
]
