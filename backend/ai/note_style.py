"""Orchestration helpers for the note_style AI stage."""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from backend.ai.manifest import ensure_note_style_section
from backend.ai.note_style_stage import (
    build_note_style_pack_for_account,
    discover_note_style_response_accounts,
)
from backend.ai.note_style_sender import send_note_style_packs_for_sid
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
    statuses: dict[str, Mapping[str, Any]] = {}

    for account in accounts:
        result = build_note_style_pack_for_account(
            sid_text, account.account_id, runs_root=runs_root_path
        )
        statuses[account.account_id] = dict(result)
        status_text = str(result.get("status") or "").lower()
        if status_text == "completed":
            built += 1
        elif status_text.startswith("skipped"):
            skipped += 1

    processed: list[str] = []
    if accounts:
        processed = send_note_style_packs_for_sid(
            sid_text, runs_root=runs_root_path
        )
    else:
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

    if not processed:
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
        len(processed),
        skipped,
    )

    return {
        "sid": sid_text,
        "accounts_discovered": len(accounts),
        "packs_built": built,
        "skipped": skipped,
        "processed_accounts": list(processed),
        "statuses": statuses,
    }


@dataclass
class _DebounceEntry:
    timer: threading.Timer


_DEBOUNCE_LOCK = threading.Lock()
_PENDING: dict[str, _DebounceEntry] = {}


def schedule_prepare_and_send(
    sid: str, *, runs_root: Path | str | None = None
) -> None:
    """Schedule :func:`prepare_and_send` for ``sid`` with debounce."""

    sid_text = str(sid or "").strip()
    if not sid_text:
        return

    delay = _debounce_delay_seconds()

    def _run() -> None:
        try:
            prepare_and_send(sid_text, runs_root=runs_root)
        except Exception:  # pragma: no cover - defensive logging
            log.exception("NOTE_STYLE_PREPARE_FAILED sid=%s", sid_text)
        finally:
            with _DEBOUNCE_LOCK:
                _PENDING.pop(sid_text, None)

    if delay <= 0:
        _run()
        return

    with _DEBOUNCE_LOCK:
        existing = _PENDING.pop(sid_text, None)
        if existing is not None:
            try:
                existing.timer.cancel()
            except Exception:
                pass

        timer = threading.Timer(delay, _run)
        timer.daemon = True
        timer.start()
        _PENDING[sid_text] = _DebounceEntry(timer=timer)


__all__ = ["prepare_and_send", "schedule_prepare_and_send"]

