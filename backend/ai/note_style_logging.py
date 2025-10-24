"""Helpers for writing note_style stage warnings to the run log file."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


log = logging.getLogger(__name__)


def append_note_style_warning(log_path: Path, message: str) -> None:
    """Append a warning ``message`` to ``log_path`` with a UTC timestamp."""

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    line = f"[{timestamp}] WARNING: {message.strip()}\n"

    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line)
    except OSError:
        log.warning(
            "NOTE_STYLE_LOG_WRITE_FAILED path=%s message=%s",
            log_path,
            message,
            exc_info=True,
        )


def _normalize_structured_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Mapping):
        return {
            str(key): _normalize_structured_value(sub_value)
            for key, sub_value in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_normalize_structured_value(item) for item in value]
    if isinstance(value, set):
        return sorted(_normalize_structured_value(item) for item in value)
    return str(value)


def log_structured_event(
    event: str,
    *,
    level: int = logging.INFO,
    logger: logging.Logger | None = None,
    **fields: Any,
) -> None:
    payload: dict[str, Any] = {"event": event}
    for key, value in fields.items():
        payload[key] = _normalize_structured_value(value)

    try:
        serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    except TypeError:
        fallback_payload = {
            key: str(value) if not isinstance(value, (str, int, float, bool, type(None))) else value
            for key, value in payload.items()
        }
        serialized = json.dumps(fallback_payload, ensure_ascii=False, sort_keys=True)

    target_logger = logger or log
    target_logger.log(level, serialized)


__all__ = ["append_note_style_warning", "log_structured_event"]
