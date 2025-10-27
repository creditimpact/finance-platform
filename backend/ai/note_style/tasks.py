"""Celery tasks for the note_style AI stage."""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, Mapping

from backend.api.tasks import app as celery
from backend.ai.manifest import ensure_note_style_section
from backend.ai.note_style import prepare_and_send
from backend.ai.note_style_logging import (
    append_note_style_warning,
    log_structured_event,
)
from backend.ai.note_style_results import record_note_style_failure
from backend.ai.note_style_sender import (
    send_note_style_pack_for_account,
    send_note_style_packs_for_sid,
)
from backend.core.ai.paths import ensure_note_style_paths


log = logging.getLogger(__name__)


def _note_style_has_packs(
    sid: str, runs_root: str | Path | None = None
) -> bool:
    runs_root_path = _resolve_runs_root(runs_root)
    base = runs_root_path / sid / "ai_packs" / "note_style" / "packs"
    if not base.is_dir():
        return False
    return any(child.suffix == ".jsonl" for child in base.iterdir())


def _resolve_runs_root(runs_root: str | Path | None) -> Path:
    if runs_root is None:
        env_value = os.getenv("RUNS_ROOT")
        return Path(env_value) if env_value else Path("runs")
    return Path(runs_root)


def _ensure_log_path(sid: str, runs_root: str | Path | None) -> Path | None:
    try:
        runs_root_path = _resolve_runs_root(runs_root)
        ensure_note_style_section(sid, runs_root=runs_root_path)
        paths = ensure_note_style_paths(runs_root_path, sid, create=True)
    except Exception:  # pragma: no cover - defensive logging
        log.warning(
            "NOTE_STYLE_TASK_LOG_PATH_FAILED sid=%s runs_root=%s",
            sid,
            runs_root,
            exc_info=True,
        )
        return None
    return paths.log_file


def _append_task_failure_log(
    *, sid: str, runs_root: str | Path | None, message: str
) -> None:
    log_path = _ensure_log_path(sid, runs_root)
    if log_path is None:
        return
    append_note_style_warning(log_path, message)


@celery.task(name="backend.ai.note_style.tasks.note_style_prepare_and_send_task")
def note_style_prepare_and_send_task(
    sid: str, runs_root: str | Path | None = None
) -> Mapping[str, Any]:
    """Celery task wrapper around :func:`prepare_and_send`."""

    sid_text = str(sid or "").strip()
    if not _note_style_has_packs(sid_text, runs_root=runs_root):
        log.info("NOTE_STYLE_AUTO: skip_send sid=%s reason=no_note_style_packs", sid_text)
        return {
            "sid": sid_text,
            "built": 0,
            "sent": 0,
            "reason": "no_note_style_packs",
        }

    start = time.monotonic()
    log_structured_event(
        "NOTE_STYLE_CELERY_START",
        logger=log,
        task="prepare_and_send",
        sid=sid_text,
        runs_root=runs_root,
    )
    try:
        result = prepare_and_send(sid_text, runs_root=runs_root)
    except Exception:
        duration = time.monotonic() - start
        log_structured_event(
            "NOTE_STYLE_CELERY_ERROR",
            logger=log,
            task="prepare_and_send",
            sid=sid_text,
            runs_root=runs_root,
            duration_seconds=duration,
            level=logging.ERROR,
        )
        _append_task_failure_log(
            sid=sid_text,
            runs_root=runs_root,
            message=f"task=prepare_and_send sid={sid_text} status=error",
        )
        raise

    duration = time.monotonic() - start
    log_structured_event(
        "NOTE_STYLE_CELERY_SUCCESS",
        logger=log,
        task="prepare_and_send",
        sid=sid_text,
        runs_root=runs_root,
        duration_seconds=duration,
    )
    log.info(
        "NOTE_STYLE_AUTO: sent sid=%s built=%s sent=%s",
        sid_text,
        result.get("built"),
        result.get("sent"),
    )
    return result


@celery.task(name="backend.ai.note_style.tasks.note_style_send_account_task")
def note_style_send_account_task(
    sid: str, account_id: str, runs_root: str | Path | None = None
) -> Mapping[str, Any]:
    """Send a single note_style pack and ingest the result."""

    sid_text = str(sid or "").strip()
    account_text = str(account_id or "").strip()
    start = time.monotonic()
    log_structured_event(
        "NOTE_STYLE_CELERY_START",
        logger=log,
        task="send_account",
        sid=sid_text,
        account_id=account_text,
        runs_root=runs_root,
    )

    processed = False
    try:
        processed = send_note_style_pack_for_account(
            sid_text, account_text, runs_root=runs_root
        )
    except Exception:
        duration = time.monotonic() - start
        log_structured_event(
            "NOTE_STYLE_CELERY_ERROR",
            logger=log,
            task="send_account",
            sid=sid_text,
            account_id=account_text,
            runs_root=runs_root,
            duration_seconds=duration,
            level=logging.ERROR,
        )
        _append_task_failure_log(
            sid=sid_text,
            runs_root=runs_root,
            message=(
                f"task=send_account sid={sid_text} account_id={account_text} status=error"
            ),
        )
        try:
            record_note_style_failure(
                sid_text,
                account_text,
                runs_root=runs_root,
                error="celery_task_error",
            )
        except Exception:  # pragma: no cover - defensive logging
            log.warning(
                "NOTE_STYLE_ACCOUNT_TASK_FAILURE_RECORD sid=%s account_id=%s",
                sid_text,
                account_text,
                exc_info=True,
            )
        raise

    duration = time.monotonic() - start
    log_structured_event(
        "NOTE_STYLE_CELERY_SUCCESS",
        logger=log,
        task="send_account",
        sid=sid_text,
        account_id=account_text,
        runs_root=runs_root,
        processed=processed,
        duration_seconds=duration,
    )
    return {"sid": sid_text, "account_id": account_text, "processed": processed}


@celery.task(name="backend.ai.note_style.tasks.note_style_send_sid_task")
def note_style_send_sid_task(
    sid: str, runs_root: str | Path | None = None
) -> Mapping[str, Any]:
    """Send all built note_style packs for ``sid`` inside Celery."""

    sid_text = str(sid or "").strip()
    start = time.monotonic()
    log_structured_event(
        "NOTE_STYLE_CELERY_START",
        logger=log,
        task="send_sid",
        sid=sid_text,
        runs_root=runs_root,
    )

    processed: list[str]
    try:
        processed = send_note_style_packs_for_sid(sid_text, runs_root=runs_root)
    except Exception:
        duration = time.monotonic() - start
        log_structured_event(
            "NOTE_STYLE_CELERY_ERROR",
            logger=log,
            task="send_sid",
            sid=sid_text,
            runs_root=runs_root,
            duration_seconds=duration,
            level=logging.ERROR,
        )
        _append_task_failure_log(
            sid=sid_text,
            runs_root=runs_root,
            message=f"task=send_sid sid={sid_text} status=error",
        )
        raise

    duration = time.monotonic() - start
    log_structured_event(
        "NOTE_STYLE_CELERY_SUCCESS",
        logger=log,
        task="send_sid",
        sid=sid_text,
        runs_root=runs_root,
        processed_accounts=processed,
        duration_seconds=duration,
    )
    return {"sid": sid_text, "processed": processed}


__all__ = [
    "note_style_prepare_and_send_task",
    "note_style_send_account_task",
    "note_style_send_sid_task",
]
