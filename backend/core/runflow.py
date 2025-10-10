from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Tuple

import functools
import json
import os
import uuid

RUNS_ROOT = Path(os.getenv("RUNS_ROOT", "runs"))


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _atomic_write_json(path: Path, obj: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{uuid.uuid4().hex}")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _steps_path(sid: str) -> Path:
    return RUNS_ROOT / sid / "runflow_steps.json"


def _events_path(sid: str) -> Path:
    return RUNS_ROOT / sid / "runflow_events.jsonl"


def _env_enabled(name: str, default: bool = True) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


_ENABLE_INSTRUMENTATION = _env_enabled("RUNFLOW_V2_ENABLED", True)
_ENABLE_STEPS = _env_enabled("RUNFLOW_STEPS_ENABLED", True)
_ENABLE_EVENTS = _env_enabled("RUNFLOW_EVENTS_ENABLED", True)


def _disabled() -> bool:
    return not _ENABLE_INSTRUMENTATION


def _load_steps_payload(sid: str) -> dict[str, Any]:
    path = _steps_path(sid)
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {
            "sid": sid,
            "schema_version": "2.0",
            "stages": {},
            "updated_at": _utcnow_iso(),
        }
    except OSError:
        return {
            "sid": sid,
            "schema_version": "2.0",
            "stages": {},
            "updated_at": _utcnow_iso(),
        }

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        payload = {}

    if not isinstance(payload, Mapping):
        payload = {}

    stages = payload.get("stages")
    if not isinstance(stages, Mapping):
        stages = {}

    result: dict[str, Any] = {
        "sid": sid,
        "schema_version": "2.0",
        "stages": {str(key): dict(value) for key, value in stages.items() if isinstance(value, Mapping)},
        "updated_at": str(payload.get("updated_at") or _utcnow_iso()),
    }
    return result


def _dump_steps_payload(sid: str, payload: Mapping[str, Any]) -> None:
    if _disabled() or not _ENABLE_STEPS:
        return
    _atomic_write_json(_steps_path(sid), payload)


def _append_event(sid: str, row: Mapping[str, Any]) -> None:
    if _disabled() or not _ENABLE_EVENTS:
        return
    _append_jsonl(_events_path(sid), row)


def _ensure_stage_container(
    data: dict[str, Any], stage: str, started_at: str
) -> Tuple[dict[str, Any], bool]:
    stages = data.setdefault("stages", {})
    if not isinstance(stages, dict):
        stages = {}
        data["stages"] = stages

    existing = stages.get(stage)
    summary: Mapping[str, Any] | None = None
    steps: list[dict[str, Any]] | None = None
    if isinstance(existing, Mapping):
        summary = (
            existing.get("summary") if isinstance(existing.get("summary"), Mapping) else None
        )
        steps_raw = existing.get("steps")
        if isinstance(steps_raw, list):
            steps = [dict(entry) for entry in steps_raw if isinstance(entry, Mapping)]
        stage_payload = dict(existing)
        started_value = stage_payload.get("started_at")
        stage_payload["started_at"] = (
            str(started_value) if isinstance(started_value, str) else started_at
        )
        stage_payload["steps"] = steps or []
        stages[stage] = stage_payload
        return stage_payload, False

    stage_payload: dict[str, Any] = {"status": "running", "started_at": started_at, "steps": []}
    if summary:
        stage_payload["summary"] = dict(summary)
    stages[stage] = stage_payload
    return stage_payload, True


def runflow_start_stage(
    sid: str, stage: str, extra: Optional[Mapping[str, Any]] = None
) -> None:
    if _disabled():
        return

    steps_enabled = _ENABLE_STEPS
    events_enabled = _ENABLE_EVENTS
    if not (steps_enabled or events_enabled):
        return

    ts = _utcnow_iso()
    created = True
    if steps_enabled:
        data = _load_steps_payload(sid)
        stage_payload, created = _ensure_stage_container(data, stage, ts)
        stage_payload["status"] = "running"
        stage_payload.setdefault("started_at", ts)
        if not isinstance(stage_payload.get("steps"), list):
            stage_payload["steps"] = []
        if extra:
            for key, value in extra.items():
                stage_payload[str(key)] = value
        data["updated_at"] = ts
        data.setdefault("stages", {})[stage] = stage_payload
        _dump_steps_payload(sid, data)

    if events_enabled and created:
        _append_event(sid, {"ts": ts, "stage": stage, "event": "start"})


def runflow_end_stage(
    sid: str,
    stage: str,
    *,
    status: str = "success",
    summary: Optional[Mapping[str, Any]] = None,
) -> None:
    if _disabled():
        return

    steps_enabled = _ENABLE_STEPS
    events_enabled = _ENABLE_EVENTS
    if not (steps_enabled or events_enabled):
        return

    ts = _utcnow_iso()
    if steps_enabled:
        data = _load_steps_payload(sid)
        stage_payload, _ = _ensure_stage_container(data, stage, ts)
        stage_payload["status"] = status
        stage_payload["ended_at"] = ts
        if summary:
            existing_summary = stage_payload.get("summary")
            if isinstance(existing_summary, Mapping):
                merged = dict(existing_summary)
                merged.update({str(k): v for k, v in summary.items()})
            else:
                merged = {str(k): v for k, v in summary.items()}
            stage_payload["summary"] = merged
        data["updated_at"] = ts
        data.setdefault("stages", {})[stage] = stage_payload
        _dump_steps_payload(sid, data)

    if events_enabled:
        event: dict[str, Any] = {"ts": ts, "stage": stage, "event": "end", "status": status}
        if summary:
            event["summary"] = {str(k): v for k, v in summary.items()}
        _append_event(sid, event)


def runflow_step(
    sid: str,
    stage: str,
    step: str,
    *,
    status: str = "success",
    account: Optional[str] = None,
    metrics: Optional[Mapping[str, Any]] = None,
    out: Optional[Mapping[str, Any]] = None,
) -> None:
    steps_enabled = _ENABLE_STEPS
    events_enabled = _ENABLE_EVENTS

    if _disabled() or not (steps_enabled or events_enabled):
        return

    ts = _utcnow_iso()
    if steps_enabled:
        data = _load_steps_payload(sid)
        stage_payload, _ = _ensure_stage_container(data, stage, ts)

        steps_list = stage_payload.setdefault("steps", [])
        if not isinstance(steps_list, list):
            steps_list = []
            stage_payload["steps"] = steps_list

        entry: dict[str, Any] = {"name": step, "status": status, "t": ts}
        if account is not None:
            entry["account"] = account
        if metrics:
            entry["metrics"] = {str(k): v for k, v in metrics.items()}
        if out:
            entry["out"] = {str(k): v for k, v in out.items()}

        updated = False
        for existing in steps_list:
            if isinstance(existing, Mapping) and existing.get("name") == step:
                existing.clear()
                existing.update(entry)
                updated = True
                break
        if not updated:
            steps_list.append(entry)

        data["updated_at"] = ts
        data.setdefault("stages", {})[stage] = stage_payload
        _dump_steps_payload(sid, data)

    if events_enabled:
        event: dict[str, Any] = {
            "ts": ts,
            "stage": stage,
            "step": step,
            "status": status,
        }
        if account is not None:
            event["account"] = account
        if metrics:
            event["metrics"] = {str(k): v for k, v in metrics.items()}
        if out:
            event["out"] = {str(k): v for k, v in out.items()}
        _append_event(sid, event)


def runflow_step_dec(stage: str, step: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def _wrap(fn: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(fn)
        def inner(*args: Any, **kwargs: Any) -> Any:
            if _disabled():
                return fn(*args, **kwargs)

            sid: Optional[str] = None
            if "sid" in kwargs and isinstance(kwargs["sid"], str):
                sid = kwargs["sid"]
            elif args:
                candidate = args[0]
                if hasattr(candidate, "sid"):
                    sid_value = getattr(candidate, "sid")
                    if isinstance(sid_value, str):
                        sid = sid_value
                elif isinstance(candidate, Mapping):
                    sid_value = candidate.get("sid")
                    if isinstance(sid_value, str):
                        sid = sid_value

            try:
                result = fn(*args, **kwargs)
            except Exception as exc:
                if sid:
                    runflow_step(
                        sid,
                        stage,
                        step,
                        status="error",
                        out={"error": exc.__class__.__name__, "msg": str(exc)},
                    )
                    runflow_end_stage(
                        sid,
                        stage,
                        status="error",
                        summary={"error": exc.__class__.__name__},
                    )
                raise

            if sid:
                runflow_step(sid, stage, step, status="success")
            return result

        return inner

    return _wrap


__all__ = [
    "runflow_start_stage",
    "runflow_end_stage",
    "runflow_step",
    "runflow_step_dec",
]
