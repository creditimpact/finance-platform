"""Celery tasks for the note_style AI stage."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Mapping

from backend.api.tasks import app as celery
from backend.ai.note_style import prepare_and_send
from backend.ai.note_style_sender import (
    send_note_style_pack_for_account,
    send_note_style_packs_for_sid,
)


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


@celery.task(name="backend.ai.note_style.note_style_send_account_task")
def note_style_send_account_task(
    sid: str, account_id: str, runs_root: str | Path | None = None
) -> Mapping[str, Any]:
    """Send a single note_style pack and ingest the result."""

    sid_text = str(sid or "").strip()
    account_text = str(account_id or "").strip()
    start = time.monotonic()
    log.info(
        "NOTE_STYLE_ACCOUNT_TASK_START sid=%s account_id=%s",
        sid_text,
        account_text,
    )

    processed = False
    try:
        processed = send_note_style_pack_for_account(
            sid_text, account_text, runs_root=runs_root
        )
    except Exception:
        duration = time.monotonic() - start
        log.exception(
            "NOTE_STYLE_ACCOUNT_TASK_ERROR sid=%s account_id=%s duration=%.3f",
            sid_text,
            account_text,
            duration,
        )
        raise

    duration = time.monotonic() - start
    log.info(
        "NOTE_STYLE_ACCOUNT_TASK_DONE sid=%s account_id=%s processed=%s duration=%.3f",
        sid_text,
        account_text,
        processed,
        duration,
    )
    return {"sid": sid_text, "account_id": account_text, "processed": processed}


@celery.task(name="backend.ai.note_style.note_style_send_sid_task")
def note_style_send_sid_task(
    sid: str, runs_root: str | Path | None = None
) -> Mapping[str, Any]:
    """Send all built note_style packs for ``sid`` inside Celery."""

    sid_text = str(sid or "").strip()
    start = time.monotonic()
    log.info("NOTE_STYLE_SEND_TASK_START sid=%s", sid_text)

    processed: list[str]
    try:
        processed = send_note_style_packs_for_sid(sid_text, runs_root=runs_root)
    except Exception:
        duration = time.monotonic() - start
        log.exception(
            "NOTE_STYLE_SEND_TASK_ERROR sid=%s duration=%.3f",
            sid_text,
            duration,
        )
        raise

    duration = time.monotonic() - start
    log.info(
        "NOTE_STYLE_SEND_TASK_DONE sid=%s processed=%s duration=%.3f",
        sid_text,
        processed,
        duration,
    )
    return {"sid": sid_text, "processed": processed}


__all__ = [
    "note_style_prepare_and_send_task",
    "note_style_send_account_task",
    "note_style_send_sid_task",
]
