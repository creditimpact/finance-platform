"""Simple runflow state machine for deciding the next pipeline action."""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Literal, Mapping, Optional, Sequence

from backend.core.io.json_io import _atomic_write_json
from backend.core.runflow import (
    _apply_umbrella_barriers,
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


def _merge_required() -> bool:
    return _env_enabled("MERGE_REQUIRED", True)


def _umbrella_require_merge() -> bool:
    return _env_enabled("UMBRELLA_REQUIRE_MERGE", True)


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

    stages_snapshot = data.get("stages")
    existing_stage: Mapping[str, Any] | None
    if isinstance(stages_snapshot, Mapping):
        existing_stage = stages_snapshot.get(stage)
    else:
        existing_stage = None

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

    timestamp = _now_iso()
    data["updated_at"] = timestamp

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

    if stage == "merge":
        result_files_value = _coerce_int(summary_payload.get("result_files"))
        if result_files_value is None:
            stage_result_files = _coerce_int(stage_payload.get("result_files"))
            if stage_result_files is None and isinstance(existing_stage, Mapping):
                stage_result_files = _coerce_int(existing_stage.get("result_files"))
            if stage_result_files is not None:
                summary_payload["result_files"] = stage_result_files

    if isinstance(existing_stage, Mapping):
        existing_summary = existing_stage.get("summary")
        if isinstance(existing_summary, Mapping):
            merged_summary = dict(existing_summary)
            merged_summary.update(summary_payload)
            summary_payload = merged_summary

    stage_status_override: Optional[str] = None
    if summary_payload:
        stage_payload["summary"] = dict(summary_payload)

    if stage == "frontend" and status != "error":
        packs_value: Optional[int] = None
        for key in ("packs_count", "packs"):
            packs_value = normalized_counts.get(key)
            if packs_value is not None:
                break
        if packs_value == 0:
            stage_status_override = "empty"

    barrier_event: Optional[dict[str, Any]] = None
    umbrella_ready_value: Optional[bool] = None
    barrier_result = _apply_umbrella_barriers(
        data,
        sid=sid,
        timestamp=timestamp,
    )
    if barrier_result is not None:
        (
            barriers_payload,
            _merge_ready_state,
            _validation_ready_state,
            _review_ready_state,
            all_ready_state,
            _barrier_ts,
        ) = barrier_result
        barrier_event = dict(barriers_payload)
        umbrella_ready_value = bool(all_ready_state)
    else:
        existing_barriers_payload = data.get("umbrella_barriers")
        if isinstance(existing_barriers_payload, Mapping):
            barrier_event = dict(existing_barriers_payload)
        umbrella_ready_existing = data.get("umbrella_ready")
        if isinstance(umbrella_ready_existing, bool):
            umbrella_ready_value = umbrella_ready_existing

    _atomic_write_json(path, data)

    runflow_end_stage(
        sid,
        stage,
        status=status,
        summary=summary_payload if summary_payload else None,
        stage_status=stage_status_override,
        empty_ok=empty_ok,
        barriers=barrier_event,
        umbrella_ready=umbrella_ready_value,
        refresh_barriers=False,
    )

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

    expected_candidates: list[int] = []
    existing_created = _maybe_int(totals.get("created_packs"))
    if existing_created is not None:
        expected_candidates.append(existing_created)

    fallback_created = _maybe_int(index_payload.get("created_packs"))
    if fallback_created is not None:
        expected_candidates.append(fallback_created)

    if isinstance(pairs_payload, list):
        expected_candidates.append(len(pairs_payload))

    expected_total: Optional[int]
    if expected_candidates:
        expected_total = max(expected_candidates)
    else:
        expected_total = None

    ready_counts_match = result_files_total == pack_files_total
    if expected_total is not None:
        ready_counts_match = ready_counts_match and result_files_total == expected_total

    if not ready_counts_match:
        raise RuntimeError(
            "merge stage artifacts not ready: results=%s packs=%s expected=%s"
            % (result_files_total, pack_files_total, expected_total)
        )

    created_packs_value = result_files_total
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

        if normalized_status in {"failed", "error"}:
            failed += 1
            ready = False
            continue

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

    if total == 0:
        return (0, 0, 0, True)

    ready = ready and completed == total
    return (total, completed, failed, ready)


def _validation_stage_ready(
    run_dir: Path, validation_stage: Mapping[str, Any] | None
) -> bool:
    total, completed, failed, ready = _validation_results_progress(run_dir)

    if total > 0:
        return ready and completed == total

    if ready:
        return True

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
    if required == 0:
        ready = True
    return (required, answered, ready)


def _review_responses_ready(run_dir: Path) -> bool:
    _, _, ready = _frontend_responses_progress(run_dir)
    return ready


def _stage_empty_ok(stage_info: Mapping[str, Any] | None) -> bool:
    if not isinstance(stage_info, Mapping):
        return False

    empty_ok_value = stage_info.get("empty_ok")
    if isinstance(empty_ok_value, bool):
        return empty_ok_value
    if isinstance(empty_ok_value, (int, float)):
        return bool(empty_ok_value)

    summary_payload = stage_info.get("summary")
    if isinstance(summary_payload, Mapping):
        summary_empty = summary_payload.get("empty_ok")
        if isinstance(summary_empty, bool):
            return summary_empty
        if isinstance(summary_empty, (int, float)):
            return bool(summary_empty)

    return False


def _merge_artifacts_progress(
    run_dir: Path,
) -> tuple[int, int, Optional[int], bool]:
    merge_dir = run_dir / "ai_packs" / "merge"
    results_dir = merge_dir / "results"
    packs_dir = merge_dir / "packs"

    try:
        result_files_total = sum(
            1 for path in results_dir.rglob("*.result.json") if path.is_file()
        )
    except OSError:
        result_files_total = 0

    try:
        pack_files_total = sum(
            1 for path in packs_dir.glob("pair_*.jsonl") if path.is_file()
        )
    except OSError:
        pack_files_total = 0

    expected_total: Optional[int] = None
    index_path = merge_dir / "pairs_index.json"
    index_payload = _load_json_mapping(index_path)
    if isinstance(index_payload, Mapping):
        totals_payload = index_payload.get("totals")
        if isinstance(totals_payload, Mapping):
            candidates = [
                _coerce_int(totals_payload.get(key))
                for key in ("created_packs", "packs_built", "total_packs")
            ]
            expected_candidates = [value for value in candidates if value is not None]
            if expected_candidates:
                expected_total = max(expected_candidates)

        if expected_total is None:
            fallback_candidates = [
                _coerce_int(index_payload.get(key))
                for key in ("created_packs", "packs_built", "pack_count")
            ]
            fallback_values = [value for value in fallback_candidates if value is not None]
            if fallback_values:
                expected_total = max(fallback_values)

        if expected_total is None:
            pairs_payload = index_payload.get("pairs")
            if isinstance(pairs_payload, Sequence):
                expected_total = len(pairs_payload)

    ready = False

    if expected_total == 0:
        ready = True
    else:
        ready = result_files_total == pack_files_total
        if expected_total is not None:
            ready = ready and result_files_total == expected_total

    if (
        not ready
        and result_files_total == 0
        and pack_files_total == 0
        and expected_total is None
        and not _merge_required()
    ):
        ready = True

    return (result_files_total, pack_files_total, expected_total, ready)


def _compute_umbrella_barriers(run_dir: Path) -> dict[str, bool]:
    runflow_path = run_dir / "runflow.json"
    runflow_payload = _load_json_mapping(runflow_path)
    stages_payload = (
        runflow_payload.get("stages")
        if isinstance(runflow_payload, Mapping)
        else None
    )

    if isinstance(stages_payload, Mapping):
        stages = stages_payload
    else:
        stages = {}

    def _stage_mapping(stage_name: str) -> Mapping[str, Any] | None:
        candidate = stages.get(stage_name)
        if isinstance(candidate, Mapping):
            return candidate
        return None

    def _stage_status_success(stage_info: Mapping[str, Any] | None) -> bool:
        if not isinstance(stage_info, Mapping):
            return False
        status_value = stage_info.get("status")
        if not isinstance(status_value, str):
            return False
        return status_value.strip().lower() == "success"

    def _summary_value(stage_info: Mapping[str, Any], key: str) -> Optional[int]:
        summary = stage_info.get("summary")
        if isinstance(summary, Mapping):
            value = _coerce_int(summary.get(key))
            if value is not None:
                return value
        return None

    merge_stage = _stage_mapping("merge")
    has_merge_stage = isinstance(merge_stage, Mapping)
    merge_empty_ok = _stage_empty_ok(merge_stage)
    merge_stage_result_files: Optional[int] = None
    merge_ready = False
    if _stage_status_success(merge_stage):
        result_files = _summary_value(merge_stage, "result_files")
        if result_files is None and isinstance(merge_stage, Mapping):
            result_files = _coerce_int(merge_stage.get("result_files"))
        if result_files is None and merge_empty_ok:
            result_files = 0
        merge_stage_result_files = result_files
        if merge_empty_ok:
            merge_ready = True
        elif result_files is not None:
            merge_ready = result_files >= 1

    ( 
        merge_disk_result_files,
        _merge_disk_pack_files,
        _merge_expected,
        merge_ready_disk,
    ) = _merge_artifacts_progress(run_dir)
    if merge_ready_disk:
        merge_ready = True
    elif merge_ready:
        if merge_stage_result_files is None:
            merge_ready = merge_empty_ok and merge_disk_result_files == 0
        else:
            merge_ready = merge_stage_result_files == merge_disk_result_files

    if not _umbrella_require_merge():
        if merge_stage_result_files is None:
            merge_result_files_for_policy = merge_disk_result_files
        else:
            merge_result_files_for_policy = merge_stage_result_files

        reason: Optional[str] = None
        if not has_merge_stage:
            reason = "no_merge_stage"
        elif merge_result_files_for_policy == 0:
            reason = "empty_merge_results"

        if reason is not None:
            log.info(
                "UMBRELLA_MERGE_OPTIONAL sid=%s reason=%s was_ready=%s merge_files=%s require_merge=0",
                run_dir.name,
                reason,
                merge_ready,
                merge_result_files_for_policy,
            )
            merge_ready = True

    validation_stage = _stage_mapping("validation")
    validation_total, validation_completed, _validation_failed, validation_ready_disk = (
        _validation_results_progress(run_dir)
    )
    if validation_total > 0:
        validation_ready_disk = validation_ready_disk and (
            validation_completed == validation_total
        )
    validation_ready = validation_ready_disk
    if not validation_ready and _stage_status_success(validation_stage):
        results_payload = validation_stage.get("results")
        if isinstance(results_payload, Mapping):
            completed = _coerce_int(results_payload.get("completed"))
            total = _coerce_int(results_payload.get("results_total"))
            if completed is not None and total is not None and completed == total:
                validation_ready = True
        if not validation_ready and _stage_empty_ok(validation_stage):
            validation_ready = True

    frontend_stage = _stage_mapping("frontend")
    review_required, review_received, review_ready_disk = _frontend_responses_progress(
        run_dir
    )
    has_frontend_stage = isinstance(frontend_stage, Mapping)
    review_disk_evidence = review_required > 0 or review_received > 0
    review_ready = False
    if has_frontend_stage or review_disk_evidence:
        review_ready = review_ready_disk and review_received >= review_required
    if not review_ready and _stage_status_success(frontend_stage):
        metrics_payload = frontend_stage.get("metrics")
        if isinstance(metrics_payload, Mapping):
            required = _coerce_int(metrics_payload.get("answers_required"))
            received = _coerce_int(metrics_payload.get("answers_received"))
            if required is not None and received is not None and received == required:
                review_ready = True
        if not review_ready and _stage_empty_ok(frontend_stage):
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

    if completed != total:
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

    empty_ok = total == 0
    stage_payload.setdefault("empty_ok", bool(empty_ok))

    results_payload = {
        "results_total": total,
        "completed": completed,
        "failed": failed,
    }
    stage_payload["results"] = results_payload

    metrics_payload = stage_payload.get("metrics")
    if isinstance(metrics_payload, Mapping):
        metrics_data = dict(metrics_payload)
    else:
        metrics_data = {}

    disk_counts = _stage_counts_from_disk("validation", run_dir)
    summary_counts: dict[str, int] = {}
    for key, value in disk_counts.items():
        coerced = _coerce_int(value)
        if coerced is None:
            continue
        stage_payload[key] = coerced
        summary_counts[key] = coerced

    summary_payload = stage_payload.get("summary")
    if isinstance(summary_payload, Mapping):
        summary = dict(summary_payload)
    else:
        summary = {}

    summary.update(
        {
            "results_total": total,
            "completed": completed,
            "failed": failed,
            "empty_ok": empty_ok,
        }
    )

    if summary_counts:
        summary.update(summary_counts)

    if metrics_data:
        stage_payload["metrics"] = metrics_data
        summary["metrics"] = dict(metrics_data)

    summary["results"] = dict(results_payload)
    stage_payload["summary"] = summary

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

    if answered != required:
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

    empty_ok = required == 0
    stage_payload.setdefault("empty_ok", bool(empty_ok))

    metrics_payload = stage_payload.get("metrics")
    if isinstance(metrics_payload, Mapping):
        metrics_data = dict(metrics_payload)
    else:
        metrics_data = {}
    metrics_data["answers_received"] = answered
    metrics_data["answers_required"] = required
    stage_payload["metrics"] = metrics_data
    stage_payload.pop("answers", None)

    packs_counts = _stage_counts_from_disk("frontend", run_dir)
    packs_count_value = _coerce_int(stage_payload.get("packs_count"))
    disk_packs = _coerce_int(packs_counts.get("packs_count")) if packs_counts else None
    if disk_packs is not None:
        packs_count_value = disk_packs
    if packs_count_value is not None:
        stage_payload["packs_count"] = packs_count_value

    summary_payload = stage_payload.get("summary")
    if isinstance(summary_payload, Mapping):
        summary = dict(summary_payload)
    else:
        summary = {}

    summary.update(
        {
            "answers_received": answered,
            "answers_required": required,
            "empty_ok": empty_ok,
        }
    )

    if packs_count_value is not None:
        summary["packs_count"] = packs_count_value

    if metrics_data:
        summary["metrics"] = dict(metrics_data)

    stage_payload["summary"] = summary

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
