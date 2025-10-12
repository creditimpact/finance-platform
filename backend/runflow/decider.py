"""Simple runflow state machine for deciding the next pipeline action."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Literal, Mapping, Optional

from backend.core.io.json_io import _atomic_write_json
from backend.core.runflow import runflow_end_stage

StageStatus = Literal["success", "error"]
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


def _load_runflow(path: Path, sid: str) -> dict[str, Any]:
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        data: dict[str, Any] = {
            "sid": sid,
            "run_state": "INIT",
            "stages": {},
            "updated_at": _now_iso(),
        }
        return data
    except OSError:
        log.warning("RUNFLOW_READ_FAILED sid=%s path=%s", sid, path, exc_info=True)
        return {
            "sid": sid,
            "run_state": "INIT",
            "stages": {},
            "updated_at": _now_iso(),
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

    return {
        "sid": sid,
        "run_state": run_state,
        "stages": dict(stages),
        "updated_at": str(payload.get("updated_at") or _now_iso()),
    }


def record_stage(
    sid: str,
    stage: StageName,
    *,
    status: StageStatus,
    counts: Dict[str, int],
    empty_ok: bool,
    notes: Optional[str] = None,
    runs_root: Optional[str | Path] = None,
) -> dict[str, Any]:
    """Persist ``stage`` information under ``runs/<sid>/runflow.json``."""

    path = _runflow_path(sid, runs_root)
    data = _load_runflow(path, sid)

    stage_payload: dict[str, Any] = {
        "status": status,
        "empty_ok": bool(empty_ok),
        "last_at": _now_iso(),
    }

    for key, value in (counts or {}).items():
        coerced = _coerce_int(value)
        stage_payload[str(key)] = coerced if coerced is not None else value

    if notes:
        stage_payload["notes"] = str(notes)

    stages = data.setdefault("stages", {})
    if not isinstance(stages, dict):
        stages = {}
        data["stages"] = stages
    stages[stage] = stage_payload

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
        dict(counts or {}),
        empty_ok,
    )

    summary_payload: dict[str, Any] = {str(key): value for key, value in (counts or {}).items()}
    summary_payload["empty_ok"] = bool(empty_ok)
    if notes:
        summary_payload["notes"] = str(notes)

    stage_status_override: Optional[str] = None
    if stage == "frontend" and status != "error":
        packs_value: Optional[int] = None
        for key in ("packs_count", "packs"):
            packs_value = _coerce_int(counts.get(key))
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
    return data


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
                findings_count = _coerce_int(validation_stage.get("findings_count"))
                if findings_count is None:
                    findings_count = 0

                if validation_status == "error":
                    _set("stop_error", "validation_error", "ERROR")
                elif validation_status != "success":
                    _set("run_validation", "validation_pending", "VALIDATING")
                elif findings_count <= 0:
                    _set("complete_no_action", "validation_no_findings", "COMPLETE_NO_ACTION")
                else:
                    frontend_stage = stages.get("frontend")
                    if not isinstance(frontend_stage, Mapping):
                        _set("gen_frontend_packs", "validation_has_findings", "VALIDATING")
                    else:
                        frontend_status = str(frontend_stage.get("status") or "")
                        if frontend_status == "error":
                            _set("stop_error", "frontend_error", "ERROR")
                        elif frontend_status == "success":
                            packs_count = _coerce_int(frontend_stage.get("packs_count")) or 0
                            if packs_count <= 0:
                                _set(
                                    "complete_no_action",
                                    "frontend_no_packs",
                                    "COMPLETE_NO_ACTION",
                                )
                            else:
                                _set(
                                    "await_input",
                                    "frontend_completed",
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

    return {"next": next_action, "reason": reason}


__all__ = ["record_stage", "decide_next", "StageStatus", "RunState"]
