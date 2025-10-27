"""Helpers for coordinating umbrella barrier side-effects."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Mapping

from backend import config
from backend.pipeline.runs import RunManifest

log = logging.getLogger(__name__)


def _stage_completed(status: Mapping[str, object] | None) -> bool:
    if not isinstance(status, Mapping):
        return False

    if bool(status.get("sent")):
        return True

    completed_at = status.get("completed_at")
    if isinstance(completed_at, str) and completed_at.strip():
        return True

    return False


def _note_style_already_sent(status: Mapping[str, object] | None) -> bool:
    if not isinstance(status, Mapping):
        return False

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
        return

    if not config.NOTE_STYLE_ENABLED:
        log.info("NOTE_STYLE_AUTOSEND_DISABLED sid=%s reason=feature", sid_text)
        return

    if not _env_flag_enabled("NOTE_STYLE_AUTOSEND", default=True):
        log.info("NOTE_STYLE_AUTOSEND_DISABLED sid=%s reason=env", sid_text)
        return

    if not _env_flag_enabled("NOTE_STYLE_STAGE_AUTORUN", default=True):
        log.info("NOTE_STYLE_STAGE_AUTORUN_DISABLED sid=%s", sid_text)
        return

    if not _env_flag_enabled("NOTE_STYLE_SEND_ON_RESPONSE_WRITE", default=True):
        log.info("NOTE_STYLE_SEND_ON_RESPONSE_WRITE_DISABLED sid=%s", sid_text)
        return

    run_dir_path = Path(run_dir)
    runs_root_path = run_dir_path.parent

    try:
        manifest = RunManifest.load_or_create(run_dir_path / "manifest.json", sid_text)
    except Exception:  # pragma: no cover - defensive logging
        log.debug(
            "NOTE_STYLE_AUTOSEND_MANIFEST_LOAD_FAILED sid=%s path=%s",
            sid_text,
            run_dir_path,
            exc_info=True,
        )
        return

    validation_status = manifest.get_ai_stage_status("validation")
    if not _stage_completed(validation_status):
        return

    note_style_status = manifest.get_ai_stage_status("note_style")
    if _note_style_already_sent(note_style_status):
        return

    try:
        from backend.runflow.manifest import resolve_note_style_stage_paths

        paths = resolve_note_style_stage_paths(runs_root_path, sid_text, create=False)
    except Exception:  # pragma: no cover - defensive logging
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
    except Exception:  # pragma: no cover - defensive logging
        log.warning(
            "NOTE_STYLE_AUTOSEND_IMPORT_FAILED sid=%s", sid_text, exc_info=True
        )
        return

    pack_count = _count_note_style_packs(paths.packs_dir)

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
    except Exception:  # pragma: no cover - defensive logging
        log.warning(
            "NOTE_STYLE_AUTOSEND_SCHEDULE_FAILED sid=%s packs=%s",
            sid_text,
            pack_count,
            exc_info=True,
        )
        return

    log.info(
        "NOTE_STYLE_AUTOSEND_AFTER_VALIDATION sid=%s packs=%s",
        sid_text,
        pack_count,
    )


__all__ = ["schedule_note_style_after_validation"]
