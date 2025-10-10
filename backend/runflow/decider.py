"""Simple runflow state machine for deciding the next pipeline action."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Literal, Mapping, Optional

from backend.core.io.json_io import _atomic_write_json

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

    _atomic_write_json(path, data)
    return data


def decide_next(sid: str, runs_root: Optional[str | Path] = None) -> dict[str, str]:
    """Return the next pipeline action for ``sid`` based on recorded stages."""

    path = _runflow_path(sid, runs_root)
    data = _load_runflow(path, sid)
    stages: Mapping[str, Mapping[str, Any]] = data.get("stages", {})  # type: ignore[assignment]

    next_action = "run_validation"
    reason = "validation_pending"
    new_state: RunState = data.get("run_state", "INIT")  # type: ignore[assignment]

    for stage_name, stage_info in stages.items():
        status = str(stage_info.get("status") or "")
        if status == "error":
            next_action = "stop_error"
            reason = f"{stage_name}_error"
            new_state = "ERROR"
            break
    else:
        merge_stage = stages.get("merge")
        total_accounts = _coerce_int(merge_stage.get("count")) if isinstance(merge_stage, Mapping) else None
        if total_accounts == 0:
            next_action = "complete_no_action"
            reason = "no_accounts"
            new_state = "COMPLETE_NO_ACTION"
        else:
            validation_stage = stages.get("validation")
            validation_status = ""
            findings_count: Optional[int] = None
            empty_ok = False
            if isinstance(validation_stage, Mapping):
                validation_status = str(validation_stage.get("status") or "")
                findings_count = _coerce_int(validation_stage.get("findings_count"))
                if findings_count is None and "findings_count" in validation_stage:
                    findings_count = 0
                empty_ok = bool(validation_stage.get("empty_ok"))

            frontend_stage = stages.get("frontend")
            frontend_status = ""
            if isinstance(frontend_stage, Mapping):
                frontend_status = str(frontend_stage.get("status") or "")

            if findings_count is not None and findings_count > 0:
                if frontend_status == "success":
                    next_action = "await_input"
                    reason = "frontend_completed"
                    new_state = "AWAITING_CUSTOMER_INPUT"
                else:
                    next_action = "gen_frontend_packs"
                    reason = "validation_has_findings"
                    new_state = "VALIDATING"
            elif validation_status != "success":
                next_action = "run_validation"
                if validation_status == "error":
                    next_action = "stop_error"
                    reason = "validation_error"
                    new_state = "ERROR"
                else:
                    reason = "validation_pending"
                    new_state = "VALIDATING"
            elif (findings_count in (0, None)) and empty_ok:
                next_action = "complete_no_action"
                reason = "validation_empty_ok"
                new_state = "COMPLETE_NO_ACTION"
            else:
                if frontend_status == "success":
                    reason = "frontend_completed"
                else:
                    reason = "validation_complete"
                next_action = "await_input"
                new_state = "AWAITING_CUSTOMER_INPUT"

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
