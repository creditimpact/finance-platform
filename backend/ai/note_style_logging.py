"""Helpers for writing note_style stage warnings to the run log file."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path


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


__all__ = ["append_note_style_warning"]
