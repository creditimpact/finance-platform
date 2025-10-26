"""Celery tasks for the note_style AI stage."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Mapping

from backend.api.tasks import app as celery
from backend.ai.note_style import prepare_and_send


log = logging.getLogger(__name__)


@celery.task(name="backend.ai.note_style.note_style_prepare_and_send_task")
def note_style_prepare_and_send_task(
    sid: str, runs_root: str | Path | None = None
) -> Mapping[str, Any]:
    """Celery task wrapper around :func:`prepare_and_send`."""

    start = time.monotonic()
    sid_text = str(sid or "").strip()
    log.info("NOTE_STYLE_TASK_START sid=%s", sid_text)
    try:
        result = prepare_and_send(sid_text, runs_root=runs_root)
    except Exception:
        duration = time.monotonic() - start
        log.exception(
            "NOTE_STYLE_TASK_ERROR sid=%s duration=%.3f", sid_text, duration
        )
        raise

    duration = time.monotonic() - start
    log.info("NOTE_STYLE_TASK_DONE sid=%s duration=%.3f", sid_text, duration)
    return result


__all__ = ["note_style_prepare_and_send_task"]
