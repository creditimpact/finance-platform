from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

import functools
import json
import os

from backend.core.runflow_steps import (
    RUNS_ROOT,
    steps_append,
    steps_init,
    steps_stage_finish,
    steps_stage_start,
)


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
    mode = 0o644
    fd = os.open(path, flags, mode)
    try:
        os.write(fd, payload)
    finally:
        os.close(fd)


def _events_path(sid: str) -> Path:
    return RUNS_ROOT / sid / "runflow_events.jsonl"


def _env_enabled(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"", "0", "false", "no", "off"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw.strip())
    except (TypeError, ValueError):
        return default
    return value


_ENABLE_STEPS = _env_enabled("RUNFLOW_VERBOSE")
_ENABLE_EVENTS = _env_enabled("RUNFLOW_EVENTS")
_STEP_SAMPLE_EVERY = max(_env_int("RUNFLOW_STEP_LOG_EVERY", 1), 1)
_PAIR_TOPN = max(_env_int("RUNFLOW_STEPS_PAIR_TOPN", 5), 0)
_ENABLE_SPANS = _env_enabled("RUNFLOW_STEPS_ENABLE_SPANS", True)


_STEP_CALL_COUNTS: dict[tuple[str, str, str, str], int] = defaultdict(int)
_STARTED_STAGES: set[tuple[str, str]] = set()


def _append_event(sid: str, row: Mapping[str, Any]) -> None:
    if not _ENABLE_EVENTS:
        return
    _append_jsonl(_events_path(sid), row)


def steps_pair_topn() -> int:
    """Return the configured Top-N threshold for merge pair steps."""

    return _PAIR_TOPN


def _reset_step_counters(sid: str, stage: str) -> None:
    if not _STEP_CALL_COUNTS:
        return
    keys_to_delete = [key for key in _STEP_CALL_COUNTS if key[0] == sid and key[1] == stage]
    for key in keys_to_delete:
        del _STEP_CALL_COUNTS[key]


def _should_record_step(
    sid: str, stage: str, substage: str, step: str, status: str
) -> bool:
    if status != "success":
        return True
    if _STEP_SAMPLE_EVERY <= 1:
        key = (sid, stage, substage, step)
        _STEP_CALL_COUNTS[key] += 1
        return True

    key = (sid, stage, substage, step)
    _STEP_CALL_COUNTS[key] += 1
    count = _STEP_CALL_COUNTS[key]
    return count == 1 or count % _STEP_SAMPLE_EVERY == 0


def _shorten_message(message: str, *, limit: int = 200) -> str:
    compact = " ".join(message.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1] + "\u2026"


def _format_error_payload(exc: Exception, where: str) -> dict[str, str]:
    message = str(exc) or exc.__class__.__name__
    short = _shorten_message(message)
    return {
        "type": exc.__class__.__name__,
        "message": short,
        "where": where,
        "hint": "see runflow_events.jsonl",
    }


def runflow_start_stage(
    sid: str, stage: str, extra: Optional[Mapping[str, Any]] = None
) -> None:
    steps_enabled = _ENABLE_STEPS
    events_enabled = _ENABLE_EVENTS
    if not (steps_enabled or events_enabled):
        return

    ts = _utcnow_iso()
    key = (sid, stage)
    created = False
    _reset_step_counters(sid, stage)
    if steps_enabled:
        steps_init(sid)
        created = steps_stage_start(sid, stage, started_at=ts, extra=extra)
    else:
        created = key not in _STARTED_STAGES

    if events_enabled and created:
        _append_event(sid, {"ts": ts, "stage": stage, "event": "start"})

    if created:
        _STARTED_STAGES.add(key)


def runflow_end_stage(
    sid: str,
    stage: str,
    *,
    status: str = "success",
    summary: Optional[Mapping[str, Any]] = None,
    stage_status: Optional[str] = None,
    empty_ok: bool = False,
) -> None:
    steps_enabled = _ENABLE_STEPS
    events_enabled = _ENABLE_EVENTS
    if not (steps_enabled or events_enabled):
        return

    ts = _utcnow_iso()
    if steps_enabled:
        status_for_steps = stage_status or status
        steps_stage_finish(
            sid,
            stage,
            status_for_steps,
            summary,
            ended_at=ts,
            empty_ok=empty_ok,
        )

    if events_enabled:
        event: dict[str, Any] = {"ts": ts, "stage": stage, "event": "end", "status": status}
        if summary:
            event["summary"] = {str(k): v for k, v in summary.items()}
        _append_event(sid, event)

    _STARTED_STAGES.discard((sid, stage))
    _reset_step_counters(sid, stage)


def runflow_event(
    sid: str,
    stage: str,
    step: str,
    *,
    status: str = "success",
    account: Optional[str] = None,
    metrics: Optional[Mapping[str, Any]] = None,
    out: Optional[Mapping[str, Any]] = None,
    substage: Optional[str] = None,
    reason: Optional[str] = None,
    span_id: Optional[str] = None,
    parent_span_id: Optional[str] = None,
    error: Optional[Mapping[str, Any]] = None,
) -> None:
    """Emit a runflow event without recording a step entry."""

    if not _ENABLE_EVENTS:
        return

    ts = _utcnow_iso()
    substage_name = substage or "default"
    event: dict[str, Any] = {
        "ts": ts,
        "stage": stage,
        "step": step,
        "status": status,
        "substage": substage_name,
    }
    if account is not None:
        event["account"] = account
    if metrics:
        event["metrics"] = {str(k): v for k, v in metrics.items()}
    if out:
        event["out"] = {str(k): v for k, v in out.items()}
    if reason is not None:
        event["reason"] = reason
    if span_id is not None:
        event["span_id"] = span_id
    if parent_span_id is not None:
        event["parent_span_id"] = parent_span_id
    if error:
        event["error"] = {str(k): v for k, v in error.items()}
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
    substage: Optional[str] = None,
    reason: Optional[str] = None,
    span_id: Optional[str] = None,
    parent_span_id: Optional[str] = None,
    error: Optional[Mapping[str, Any]] = None,
) -> None:
    steps_enabled = _ENABLE_STEPS
    events_enabled = _ENABLE_EVENTS

    if not (steps_enabled or events_enabled):
        return

    ts = _utcnow_iso()
    substage_name = substage or "default"

    if status == "success":
        should_record = _should_record_step(sid, stage, substage_name, step, status)
        if not should_record:
            return
    else:
        # Ensure counters stay up to date even when not sampling success entries.
        _STEP_CALL_COUNTS[(sid, stage, substage_name, step)] += 1

    if steps_enabled:
        step_span_id = span_id if _ENABLE_SPANS else None
        step_parent_span_id = parent_span_id if _ENABLE_SPANS else None
        should_write_step = True
        if stage == "merge":
            allowed_success_steps = {
                "pack_create",
                "acctnum_match_level",
                "acctnum_pairs_summary",
                "no_merge_candidates",
                "load_cases",
                "score_pairs",
            }
            if status == "success":
                should_write_step = step in allowed_success_steps
            else:
                should_write_step = False
        if should_write_step:
            steps_append(
                sid,
                stage,
                step,
                status,
                t=ts,
                account=account,
                metrics=metrics,
                out=out,
                reason=reason,
                span_id=step_span_id,
                parent_span_id=step_parent_span_id,
                error=error,
            )
        elif stage == "merge":
            steps_stage_start(sid, stage, started_at=ts)

    if events_enabled:
        event: dict[str, Any] = {
            "ts": ts,
            "stage": stage,
            "step": step,
            "status": status,
            "substage": substage_name,
        }
        if account is not None:
            event["account"] = account
        if metrics:
            event["metrics"] = {str(k): v for k, v in metrics.items()}
        if out:
            event["out"] = {str(k): v for k, v in out.items()}
        if reason is not None:
            event["reason"] = reason
        if span_id is not None:
            event["span_id"] = span_id
        if parent_span_id is not None:
            event["parent_span_id"] = parent_span_id
        if error:
            event["error"] = {str(k): v for k, v in error.items()}
        _append_event(sid, event)


def runflow_step_dec(stage: str, step: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def _wrap(fn: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(fn)
        def inner(*args: Any, **kwargs: Any) -> Any:
            if not (_ENABLE_STEPS or _ENABLE_EVENTS):
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

            where = f"{fn.__module__}:{getattr(fn, '__qualname__', fn.__name__)}"

            try:
                result = fn(*args, **kwargs)
            except Exception as exc:
                if sid:
                    error_payload = _format_error_payload(exc, where)
                    runflow_step(
                        sid,
                        stage,
                        step,
                        status="error",
                        error=error_payload,
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
    "runflow_event",
    "runflow_step",
    "runflow_step_dec",
    "steps_pair_topn",
]
