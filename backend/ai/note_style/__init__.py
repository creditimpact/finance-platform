"""Orchestration helpers for the note_style AI stage."""

from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Mapping

from backend import config
from backend.ai.manifest import ensure_note_style_section
from backend.ai.note_style_stage import (
    build_note_style_pack_for_account,
    discover_note_style_response_accounts,
)
from backend.core.runflow import runflow_barriers_refresh
from backend.runflow.decider import record_stage, reconcile_umbrella_barriers


log = logging.getLogger(__name__)

_DEBOUNCE_MS_ENV = "NOTE_STYLE_DEBOUNCE_MS"
_DEFAULT_DEBOUNCE_MS = 750


def _resolve_runs_root(runs_root: Path | str | None) -> Path:
    if runs_root is None:
        env_value = os.getenv("RUNS_ROOT")
        return Path(env_value) if env_value else Path("runs")
    return Path(runs_root)


def _debounce_delay_seconds() -> float:
    raw = os.getenv(_DEBOUNCE_MS_ENV)
    if raw is None:
        return _DEFAULT_DEBOUNCE_MS / 1000.0
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return _DEFAULT_DEBOUNCE_MS / 1000.0
    if value <= 0:
        return 0.0
    return value / 1000.0


def prepare_and_send(
    sid: str, *, runs_root: Path | str | None = None
) -> Mapping[str, Any]:
    """Discover, build, and send note_style packs for ``sid``."""

    sid_text = str(sid or "").strip()
    if not sid_text:
        raise ValueError("sid is required")

    if not config.NOTE_STYLE_ENABLED:
        log.info("NOTE_STYLE_DISABLED sid=%s", sid_text)
        return {
            "sid": sid_text,
            "accounts_discovered": 0,
            "packs_built": 0,
            "skipped": 0,
            "processed_accounts": [],
            "statuses": {},
        }

    runs_root_path = _resolve_runs_root(runs_root)
    ensure_note_style_section(sid_text, runs_root=runs_root_path)

    accounts = discover_note_style_response_accounts(
        sid_text, runs_root=runs_root_path
    )
    log.info(
        "NOTE_STYLE_PREPARE sid=%s discovered=%s", sid_text, len(accounts)
    )

    built = 0
    skipped = 0
    scheduled: list[str] = []
    statuses: dict[str, Mapping[str, Any]] = {}

    for account in accounts:
        result = build_note_style_pack_for_account(
            sid_text, account.account_id, runs_root=runs_root_path
        )
        statuses[account.account_id] = dict(result)
        status_text = str(result.get("status") or "").lower()
        if status_text == "completed":
            built += 1
            scheduled.append(account.account_id)
        elif status_text.startswith("skipped"):
            skipped += 1

    if built > 0:
        for account_id in scheduled:
            try:
                schedule_send_for_account(
                    sid_text, account_id, runs_root=runs_root_path
                )
            except Exception:  # pragma: no cover - defensive logging
                log.warning(
                    "NOTE_STYLE_SEND_TASK_SCHEDULE_FAILED sid=%s account_id=%s",
                    sid_text,
                    account_id,
                    exc_info=True,
                )

    if not accounts:
        try:
            record_stage(
                sid_text,
                "note_style",
                status="success",
                counts={"packs_total": 0},
                empty_ok=True,
                metrics={"packs_total": 0},
                results={"results_total": 0, "completed": 0, "failed": 0},
                runs_root=runs_root_path,
            )
        except Exception:  # pragma: no cover - defensive logging
            log.warning(
                "NOTE_STYLE_PREPARE_STAGE_RECORD_FAILED sid=%s", sid_text, exc_info=True
            )

    if not built:
        try:
            runflow_barriers_refresh(sid_text)
        except Exception:  # pragma: no cover - defensive logging
            log.warning(
                "NOTE_STYLE_PREPARE_BARRIERS_REFRESH_FAILED sid=%s",
                sid_text,
                exc_info=True,
            )
        try:
            reconcile_umbrella_barriers(sid_text, runs_root=runs_root_path)
        except Exception:  # pragma: no cover - defensive logging
            log.warning(
                "NOTE_STYLE_PREPARE_BARRIERS_RECONCILE_FAILED sid=%s",
                sid_text,
                exc_info=True,
        )

    log.info(
        "NOTE_STYLE_PREPARE_DONE sid=%s discovered=%s built=%s sent=%s skipped=%s",
        sid_text,
        len(accounts),
        built,
        len(scheduled),
        skipped,
    )

    return {
        "sid": sid_text,
        "accounts_discovered": len(accounts),
        "packs_built": built,
        "skipped": skipped,
        "processed_accounts": list(scheduled),
        "statuses": statuses,
    }


def schedule_send_for_account(
    sid: str,
    account_id: str,
    *,
    runs_root: Path | str | None = None,
) -> None:
    """Enqueue a Celery task to send ``account_id``'s note_style pack."""

    sid_text = str(sid or "").strip()
    account_text = str(account_id or "").strip()
    if not sid_text or not account_text:
        return

    if not config.NOTE_STYLE_ENABLED:
        log.info(
            "NOTE_STYLE_DISABLED sid=%s account_id=%s", sid_text, account_text
        )
        return

    if runs_root is None:
        runs_root_arg: str | None = None
    else:
        try:
            runs_root_arg = os.fspath(runs_root)
        except TypeError:
            runs_root_arg = str(runs_root)

    log.info(
        "NOTE_STYLE_ACCOUNT_TASK_ENQUEUE sid=%s account_id=%s", sid_text, account_text
    )

    from backend.ai.note_style.tasks import note_style_send_account_task

    note_style_send_account_task.delay(
        sid_text, account_text, runs_root=runs_root_arg
    )


_DEBOUNCE_LOCK = threading.Lock()
_LAST_ENQUEUED: dict[str, float] = {}


def schedule_prepare_and_send(
    sid: str, *, runs_root: Path | str | None = None
) -> None:
    """Schedule :func:`prepare_and_send` for ``sid`` with debounce."""

    sid_text = str(sid or "").strip()
    if not sid_text:
        return

    if not config.NOTE_STYLE_ENABLED:
        log.info("NOTE_STYLE_DISABLED sid=%s", sid_text)
        return

    delay = _debounce_delay_seconds()
    now = time.monotonic()

    enqueue = True
    if delay > 0:
        with _DEBOUNCE_LOCK:
            last = _LAST_ENQUEUED.get(sid_text)
            if last is not None and (now - last) < delay:
                enqueue = False
            else:
                _LAST_ENQUEUED[sid_text] = now
    else:
        with _DEBOUNCE_LOCK:
            _LAST_ENQUEUED.pop(sid_text, None)

    if not enqueue:
        log.info("NOTE_STYLE_PREPARE_DEBOUNCED sid=%s", sid_text)
        return

    runs_root_arg: str | None
    if runs_root is None:
        runs_root_arg = None
    else:
        try:
            runs_root_arg = os.fspath(runs_root)
        except TypeError:
            runs_root_arg = str(runs_root)

    log.info("NOTE_STYLE_TASK_ENQUEUE sid=%s", sid_text)

    from backend.ai.note_style.tasks import note_style_prepare_and_send_task

    note_style_prepare_and_send_task.delay(sid_text, runs_root=runs_root_arg)


__all__ = ["prepare_and_send", "schedule_prepare_and_send", "schedule_send_for_account"]

