"""Helpers for coordinating umbrella barrier side-effects."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Mapping

from backend import config
from backend.pipeline.runs import RunManifest
from backend.runflow.counters import note_style_stage_counts

log = logging.getLogger(__name__)


def _safe_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _collect_note_style_metrics(run_dir_path: Path) -> tuple[int | None, int | None]:
    try:
        counters = note_style_stage_counts(run_dir_path)
    except Exception:  # pragma: no cover - defensive logging
        log.debug(
            "NOTE_STYLE_AUTOSEND_METRICS_FAILED path=%s",
            run_dir_path,
            exc_info=True,
        )
        return (None, None)

    if not counters:
        return (0, 0)

    built_total = _safe_int(counters.get("packs_total"))
    completed_total = _safe_int(counters.get("packs_completed")) or 0
    failed_total = _safe_int(counters.get("packs_failed")) or 0

    terminal_total: int | None
    if completed_total or failed_total:
        terminal_total = completed_total + failed_total
    else:
        terminal_total = 0 if built_total == 0 else None

    return (built_total, terminal_total)


def _log_autosend_decision(
    *,
    sid: str,
    reason: str,
    sent: bool | None,
    built: int | None,
    terminal: int | None,
    level: int = logging.INFO,
    error_code: str | None = None,
    error_type: str | None = None,
) -> None:
    sent_text = "unknown" if sent is None else ("true" if sent else "false")
    built_text = "unknown" if built is None else str(built)
    terminal_text = "unknown" if terminal is None else str(terminal)

    message = (
        "NOTE_STYLE_AUTOSEND_DECISION sid=%s sent=%s built=%s terminal=%s reason=%s"
        % (sid, sent_text, built_text, terminal_text, reason)
    )
    if error_code:
        message += f" error.code={error_code}"
    if error_type:
        message += f" error.type={error_type}"
    log.log(level, message)


def _stage_completed(status: Mapping[str, object] | None) -> bool:
    if not isinstance(status, Mapping):
        return False

    if bool(status.get("failed")):
        return True

    if bool(status.get("sent")):
        return True

    completed_at = status.get("completed_at")
    if isinstance(completed_at, str) and completed_at.strip():
        return True

    return False


def _note_style_already_sent(status: Mapping[str, object] | None) -> bool:
    if not isinstance(status, Mapping):
        return False

    if bool(status.get("failed")):
        return True

    if bool(status.get("sent")):
        return True

    completed_at = status.get("completed_at")
    if isinstance(completed_at, str) and completed_at.strip():
        return True

    return False


def _count_note_style_packs(packs_dir: Path) -> int:
    try:
        entries = list(packs_dir.iterdir())
    except FileNotFoundError:
        return 0
    except NotADirectoryError:
        return 0
    except OSError:
        log.debug(
            "NOTE_STYLE_PACKS_DISCOVERY_FAILED path=%s", packs_dir, exc_info=True
        )
        return 0

    total = 0
    for entry in entries:
        if not entry.is_file():
            continue
        name = entry.name
        if name.startswith("."):
            continue
        lowered = name.lower()
        if lowered.endswith(".jsonl") or lowered.endswith(".json"):
            total += 1
    return total


def _env_flag_enabled(name: str, *, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default

    lowered = raw.strip().lower()
    if lowered in {"", "0", "false", "no", "off"}:
        return False

    return True


def _normalize_runs_root_arg(runs_root: Path | None) -> str | None:
    if runs_root is None:
        return None
    try:
        return os.fspath(runs_root)
    except TypeError:
        return str(runs_root)


def schedule_note_style_after_validation(
    sid: str,
    *,
    run_dir: Path | str,
) -> None:
    """Schedule note_style autosend after validation completes for ``sid``."""

    sid_text = str(sid or "").strip()
    if not sid_text:
        _log_autosend_decision(
            sid="<missing>",
            reason="invalid_sid",
            sent=None,
            built=None,
            terminal=None,
        )
        return

    run_dir_path = Path(run_dir)
    runs_root_path = run_dir_path.parent

    metrics_cache: tuple[int | None, int | None] | None = None

    def _metrics() -> tuple[int | None, int | None]:
        nonlocal metrics_cache
        if metrics_cache is None:
            metrics_cache = _collect_note_style_metrics(run_dir_path)
        return metrics_cache

    if not config.NOTE_STYLE_ENABLED:
        built_total, terminal_total = _metrics()
        _log_autosend_decision(
            sid=sid_text,
            reason="disabled_feature",
            sent=None,
            built=built_total,
            terminal=terminal_total,
        )
        return

    if not _env_flag_enabled("NOTE_STYLE_AUTOSEND", default=True):
        built_total, terminal_total = _metrics()
        _log_autosend_decision(
            sid=sid_text,
            reason="disabled_env",
            sent=None,
            built=built_total,
            terminal=terminal_total,
        )
        return

    if not _env_flag_enabled("NOTE_STYLE_STAGE_AUTORUN", default=True):
        built_total, terminal_total = _metrics()
        _log_autosend_decision(
            sid=sid_text,
            reason="stage_autorun_disabled",
            sent=None,
            built=built_total,
            terminal=terminal_total,
        )
        return

    if not _env_flag_enabled("NOTE_STYLE_SEND_ON_RESPONSE_WRITE", default=True):
        built_total, terminal_total = _metrics()
        _log_autosend_decision(
            sid=sid_text,
            reason="send_on_write_disabled",
            sent=None,
            built=built_total,
            terminal=terminal_total,
        )
        return

    try:
        manifest = RunManifest.load_or_create(run_dir_path / "manifest.json", sid_text)
    except Exception as exc:  # pragma: no cover - defensive logging
        built_total, terminal_total = _metrics()
        _log_autosend_decision(
            sid=sid_text,
            reason="manifest_error",
            sent=None,
            built=built_total,
            terminal=terminal_total,
            level=logging.ERROR,
            error_code="manifest_load",
            error_type=type(exc).__name__,
        )
        log.debug(
            "NOTE_STYLE_AUTOSEND_MANIFEST_LOAD_FAILED sid=%s path=%s",
            sid_text,
            run_dir_path,
            exc_info=True,
        )
        return

    validation_status = manifest.get_ai_stage_status("validation")
    note_style_status = manifest.get_ai_stage_status("note_style")
    sent_flag = bool(note_style_status.get("sent"))

    if not _stage_completed(validation_status):
        built_total, terminal_total = _metrics()
        _log_autosend_decision(
            sid=sid_text,
            reason="validation_pending",
            sent=sent_flag,
            built=built_total,
            terminal=terminal_total,
        )
        return

    if _note_style_already_sent(note_style_status):
        built_total, terminal_total = _metrics()
        _log_autosend_decision(
            sid=sid_text,
            reason="already_terminal",
            sent=sent_flag,
            built=built_total,
            terminal=terminal_total,
        )
        return

    try:
        from backend.runflow.manifest import resolve_note_style_stage_paths

        paths = resolve_note_style_stage_paths(runs_root_path, sid_text, create=False)
    except Exception as exc:  # pragma: no cover - defensive logging
        built_total, terminal_total = _metrics()
        _log_autosend_decision(
            sid=sid_text,
            reason="path_error",
            sent=sent_flag,
            built=built_total,
            terminal=terminal_total,
            level=logging.ERROR,
            error_code="path_resolve",
            error_type=type(exc).__name__,
        )
        log.debug(
            "NOTE_STYLE_STAGE_PATH_RESOLVE_FAILED sid=%s runs_root=%s",
            sid_text,
            runs_root_path,
            exc_info=True,
        )
        return

    runs_root_arg = _normalize_runs_root_arg(runs_root_path)

    try:
        from backend.ai.note_style.tasks import note_style_prepare_and_send_task
    except Exception as exc:  # pragma: no cover - defensive logging
        built_total, terminal_total = _metrics()
        _log_autosend_decision(
            sid=sid_text,
            reason="task_import_error",
            sent=sent_flag,
            built=built_total,
            terminal=terminal_total,
            level=logging.ERROR,
            error_code="task_import",
            error_type=type(exc).__name__,
        )
        log.warning(
            "NOTE_STYLE_AUTOSEND_IMPORT_FAILED sid=%s", sid_text, exc_info=True
        )
        return

    pack_count = _count_note_style_packs(paths.packs_dir)
    built_total, terminal_total = _metrics()
    if built_total is None or pack_count > built_total:
        metrics_cache = (pack_count, terminal_total)
        built_total = pack_count

    log.info(
        "NOTE_STYLE_AUTOSEND_READY sid=%s packs=%s", sid_text, pack_count
    )

    try:
        if runs_root_arg is None:
            note_style_prepare_and_send_task.delay(sid_text)
        else:
            note_style_prepare_and_send_task.delay(
                sid_text, runs_root=runs_root_arg
            )
    except Exception as exc:  # pragma: no cover - defensive logging
        built_total, terminal_total = _metrics()
        _log_autosend_decision(
            sid=sid_text,
            reason="enqueue_failed",
            sent=sent_flag,
            built=built_total,
            terminal=terminal_total,
            level=logging.ERROR,
            error_code="schedule",
            error_type=type(exc).__name__,
        )
        log.warning(
            "NOTE_STYLE_AUTOSEND_SCHEDULE_FAILED sid=%s packs=%s",
            sid_text,
            pack_count,
            exc_info=True,
        )
        return

    built_total, terminal_total = _metrics()
    _log_autosend_decision(
        sid=sid_text,
        reason="enqueued",
        sent=sent_flag,
        built=built_total,
        terminal=terminal_total,
    )
    log.info(
        "NOTE_STYLE_AUTOSEND_AFTER_VALIDATION sid=%s packs=%s",
        sid_text,
        pack_count,
    )


__all__ = ["schedule_note_style_after_validation"]
