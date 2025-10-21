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
from backend.runflow.counters import stage_counts as _stage_counts_from_disk
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


def _validation_results_progress(run_dir: Path) -> tuple[int, int, int, bool]:
    """Return (total, completed, failed, ready) for validation AI results."""

    index_path = run_dir / "ai_packs" / "validation" / "index.json"
    try:
        index = load_validation_index(index_path)
    except FileNotFoundError:
        return (0, 0, 0, False)
    except Exception:  # pragma: no cover - defensive
        log.warning("RUNFLOW_VALIDATION_INDEX_LOAD_FAILED path=%s", index_path, exc_info=True)
        return (0, 0, 0, False)

    strict = _strict_validation_enabled()
    total = 0
    completed = 0
    failed = 0
    ready = True

    for record in getattr(index, "packs", ()):  # type: ignore[attr-defined]
        total += 1
        status = getattr(record, "status", "")
        normalized_status = status.strip().lower() if isinstance(status, str) else ""
        if normalized_status == "completed":
            try:
                result_path = index.resolve_result_json_path(record)  # type: ignore[attr-defined]
            except Exception:  # pragma: no cover - defensive
                log.warning(
                    "RUNFLOW_VALIDATION_RESULT_RESOLVE_FAILED account_id=%s",
                    getattr(record, "account_id", "unknown"),
                    exc_info=True,
                )
                ready = False
                continue

            try:
                exists = result_path.is_file()
            except OSError:
                exists = False

            if not exists:
                ready = False
                continue

            completed += 1
        elif normalized_status == "failed":
            failed += 1
            if strict:
                ready = False
        else:
            ready = False

    if total == 0:
        return (0, 0, 0, True)

    if strict:
        ready = ready and completed == total
    else:
        ready = ready and (completed + failed) == total

    return (total, completed, failed, ready)


def _validation_stage_ready(run_dir: Path) -> bool:
    _, _, _, ready = _validation_results_progress(run_dir)
    return ready


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
    manifest_path = run_dir / "frontend" / "review" / "index.json"
    manifest = _load_json_mapping(manifest_path)
    if manifest is None:
        return (0, 0, False)

    raw_packs = manifest.get("packs")
    account_ids: list[str] = []
    if isinstance(raw_packs, list):
        for entry in raw_packs:
            if not isinstance(entry, Mapping):
                continue
            account_id = entry.get("account_id")
            if isinstance(account_id, str) and account_id.strip():
                account_ids.append(account_id.strip())
                continue
            path_text = entry.get("path")
            if isinstance(path_text, str) and path_text.strip():
                account_ids.append(Path(path_text).stem)

    counts = manifest.get("counts") if isinstance(manifest.get("counts"), Mapping) else None
    packs_count_int = None
    responses_count_int = None
    if isinstance(counts, Mapping):
        packs_count_int = _coerce_int(counts.get("packs"))
        responses_count_int = _coerce_int(counts.get("responses"))

    if not account_ids:
        required = packs_count_int or 0
        if required <= 0:
            return (0, responses_count_int or 0, True)
        answered = responses_count_int or 0
        ready = answered >= required and required > 0
        return (required, answered, ready)

    responses_dir = run_dir / "frontend" / "review" / "responses"
    required = len(account_ids)
    explanation_required = _review_explanation_required()
    answered_ids: set[str] = set()

    for account_id in account_ids:
        filename = _response_filename_for_account(account_id)
        candidate = responses_dir / filename
        try:
            is_file = candidate.is_file()
        except OSError:
            is_file = False
        if not is_file:
            continue

        payload = _load_json_mapping(candidate)
        if payload is None:
            continue

        received_at = payload.get("received_at")
        if not isinstance(received_at, str) or not received_at.strip():
            continue

        answers = payload.get("answers")
        if not isinstance(answers, Mapping):
            continue

        if explanation_required:
            explanation = answers.get("explanation")
            if not isinstance(explanation, str) or not explanation.strip():
                continue

        answered_ids.add(account_id)

    answered = len(answered_ids)
    ready = answered >= required and required > 0
    return (required, answered, ready)


def _review_responses_ready(run_dir: Path) -> bool:
    _, _, ready = _frontend_responses_progress(run_dir)
    return ready


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

    answers_payload = {
        "answers_received": answered,
        "answers_required": required,
    }
    stage_payload["answers"] = answers_payload

    stages["frontend"] = stage_payload
    data["updated_at"] = _now_iso()

    _atomic_write_json(path, data)
    runflow_refresh_umbrella_barriers(sid)


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
    runflow_path = run_dir / "runflow.json"
    runflow_payload = _load_json_mapping(runflow_path)
    runflow_stages = (
        runflow_payload.get("stages") if isinstance(runflow_payload, Mapping) else None
    )

    steps_path = run_dir / "runflow_steps.json"
    steps_payload = _load_json_mapping(steps_path)
    steps_stages = steps_payload.get("stages") if isinstance(steps_payload, Mapping) else None

    def _combined_stage_status(stage_name: str) -> str:
        status = _stage_status(runflow_stages, stage_name)
        if status:
            return status
        return _stage_status(steps_stages, stage_name)

    merge_status = _combined_stage_status("merge")
    validation_status = _combined_stage_status("validation")

    merge_ready = merge_status in {"success", "skipped"}
    validation_ready = False
    if validation_status in {"success", "skipped"}:
        validation_ready = _validation_stage_ready(run_dir)

    review_ready = _review_responses_ready(run_dir)

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
]
