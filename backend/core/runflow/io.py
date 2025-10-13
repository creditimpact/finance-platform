from __future__ import annotations

"""Helper utilities for emitting runflow stage lifecycle records."""

from typing import Mapping, MutableMapping, Optional

from backend.core.runflow import runflow_end_stage, runflow_start_stage


def _normalize_summary(summary: Optional[Mapping[str, object]]) -> dict[str, object]:
    if not summary:
        return {}
    normalized: dict[str, object] = {}
    for key, value in summary.items():
        normalized[str(key)] = value
    return normalized


def _short_message(message: str, *, limit: int = 200) -> str:
    compact = " ".join(str(message).split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1] + "\u2026"


def _traceback_tail(payload: str, *, limit: int = 500) -> str:
    text = str(payload or "").strip()
    if len(text) <= limit:
        return text
    return text[-limit:]


def runflow_stage_start(stage: str, *, sid: str, substage: str = "default") -> None:
    """Mark ``stage`` as started for ``sid`` in runflow outputs."""

    extra: Optional[MutableMapping[str, object]]
    substage_name = substage.strip()
    if substage_name:
        extra = {"substage": substage_name}
    else:
        extra = None
    runflow_start_stage(sid, stage, extra=extra)


def runflow_stage_end(
    stage: str,
    *,
    sid: str,
    status: str = "success",
    summary: Optional[Mapping[str, object]] = None,
    empty_ok: Optional[bool] = None,
) -> None:
    """Finalize ``stage`` for ``sid`` with the provided ``summary``."""

    summary_payload = _normalize_summary(summary)
    if empty_ok:
        summary_payload.setdefault("empty_ok", True)

    stage_status = None
    if empty_ok and status != "error":
        stage_status = "empty"

    runflow_end_stage(
        sid,
        stage,
        status=status,
        summary=summary_payload or None,
        stage_status=stage_status,
        empty_ok=bool(empty_ok),
    )


def runflow_stage_error(
    stage: str,
    *,
    sid: str,
    error_type: str,
    message: str,
    traceback_tail: str,
    hint: Optional[str] = None,
    summary: Optional[Mapping[str, object]] = None,
) -> None:
    """Record an error outcome for ``stage`` on ``sid``."""

    error_payload: dict[str, object] = {
        "type": str(error_type or "Error"),
        "message": _short_message(message),
        "traceback_tail": _traceback_tail(traceback_tail),
    }
    if hint:
        error_payload["hint"] = hint

    summary_payload = _normalize_summary(summary)
    summary_payload["error"] = error_payload

    runflow_stage_end(
        stage,
        sid=sid,
        status="error",
        summary=summary_payload,
        empty_ok=False,
    )


__all__ = [
    "runflow_stage_start",
    "runflow_stage_end",
    "runflow_stage_error",
]
