"""Helpers for synchronizing runflow decisions with the run manifest."""

from __future__ import annotations

import glob
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from backend import config
from backend.ai.note_style_paths import _normalize_path_for_worker
from backend.core.paths.frontend_review import ensure_frontend_review_dirs
from backend.pipeline.runs import RUNS_ROOT_ENV, RunManifest, persist_manifest


log = logging.getLogger(__name__)


_WINDOWS_DRIVE_PATTERN = re.compile(r"^[A-Za-z]:[/\\]")


# Lazily imported to avoid circular imports during module initialization.
schedule_prepare_and_send = None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def _normalize_manifest_path_value(
    value: Path | str | None, *, run_dir: Path
) -> str | None:
    """Return a normalized string representation for manifest paths."""

    if value is None:
        return None

    try:
        text = os.fspath(value)
    except TypeError:
        return None

    sanitized = str(text).strip()
    if not sanitized:
        return None

    sanitized = sanitized.replace("\\", "/")

    if _WINDOWS_DRIVE_PATTERN.match(sanitized):
        sanitized = sanitized[2:]

    run_dir_text = str(run_dir).strip().replace("\\", "/")
    if _WINDOWS_DRIVE_PATTERN.match(run_dir_text):
        run_dir_text = run_dir_text[2:]

    run_dir_lower = run_dir_text.lower()
    sanitized_lower = sanitized.lower()
    if run_dir_lower and sanitized_lower.startswith(run_dir_lower):
        suffix = sanitized[len(run_dir_text) :]
        sanitized = run_dir_text + suffix

    return sanitized


def _resolve_manifest(
    sid: str,
    *,
    manifest: Optional[RunManifest] = None,
    runs_root: Optional[Path | str] = None,
) -> RunManifest:
    if manifest is not None:
        return manifest

    if runs_root is not None:
        if isinstance(runs_root, Path):
            base = runs_root.resolve()
        else:
            text = str(runs_root or "").strip()
            sanitized = text.replace("\\", "/")
            if len(sanitized) >= 2 and sanitized[1] == ":":
                try:
                    base = _normalize_path_for_worker(Path("/"), sanitized)
                except ValueError:
                    base = Path("runs").resolve()
            else:
                candidate = Path(sanitized)
                if candidate.is_absolute():
                    base = candidate.resolve()
                else:
                    base = (Path.cwd() / candidate).resolve()
        os.environ.setdefault(RUNS_ROOT_ENV, str(base))
        manifest_path = base / sid / "manifest.json"
        return RunManifest.load_or_create(manifest_path, sid=sid)

    return RunManifest.for_sid(sid)


def update_manifest_state(
    sid: str,
    state: str,
    *,
    manifest: Optional[RunManifest] = None,
    runs_root: Optional[Path | str] = None,
) -> RunManifest:
    """Update the manifest ``status`` field for ``sid`` to ``state``.

    Parameters
    ----------
    sid:
        The session identifier whose manifest should be updated.
    state:
        The new status string to persist into the manifest.
    manifest:
        Optional pre-loaded manifest instance. When provided, it is updated
        in-place and returned without reloading from disk.
    runs_root:
        Optional runs root override used when ``manifest`` is not supplied.
    """

    target_manifest = _resolve_manifest(
        sid, manifest=manifest, runs_root=runs_root
    )

    previous_state = str(target_manifest.data.get("run_state") or "")

    target_manifest.data["status"] = str(state)
    target_manifest.data["run_state"] = str(state)
    persist_manifest(target_manifest)

    state_text = str(state)
    if (
        state_text == "AWAITING_CUSTOMER_INPUT"
        and previous_state != "AWAITING_CUSTOMER_INPUT"
    ):
        try:
            base_dir = target_manifest.path.resolve().parent.parent
        except Exception:
            base_dir = None

        effective_runs_root = runs_root if runs_root is not None else base_dir

        if config.NOTE_STYLE_ENABLED:
            global schedule_prepare_and_send
            if schedule_prepare_and_send is None:
                from backend.ai.note_style import schedule_prepare_and_send as _schedule_prepare_and_send

                schedule_prepare_and_send = _schedule_prepare_and_send

            try:
                schedule_prepare_and_send(sid, runs_root=effective_runs_root)
            except Exception:  # pragma: no cover - defensive logging
                log.warning(
                    "NOTE_STYLE_PREPARE_SCHEDULE_STATE_FAILED sid=%s state=%s",
                    sid,
                    state_text,
                    exc_info=True,
                )

    return target_manifest


def update_manifest_frontend(
    sid: str,
    *,
    packs_dir: Optional[Path | str],
    packs_count: int,
    built: bool,
    last_built_at: Optional[str],
    manifest: Optional[RunManifest] = None,
    runs_root: Optional[Path | str] = None,
) -> RunManifest:
    target_manifest = _resolve_manifest(
        sid, manifest=manifest, runs_root=runs_root
    )

    run_dir = target_manifest.path.parent
    canonical_paths = ensure_frontend_review_dirs(str(run_dir))

    run_dir_path = run_dir.resolve()

    packs_dir_path = canonical_paths["packs_dir"]
    responses_dir_path = canonical_paths["responses_dir"]
    review_dir_path = canonical_paths["review_dir"]
    frontend_base = canonical_paths["frontend_base"]
    index_path = canonical_paths["index"]
    legacy_index_path = canonical_paths.get("legacy_index")

    packs_count_glob = len(glob.glob(os.path.join(packs_dir_path, "idx-*.json")))
    packs_count_param = int(packs_count or 0)
    packs_count_value = max(packs_count_glob, packs_count_param)

    responses_count = len(glob.glob(os.path.join(responses_dir_path, "*.json")))
    now_iso = _now_iso()

    last_built_value: str | None
    if built:
        last_built_value = (
            str(last_built_at) if last_built_at else now_iso
        )
    else:
        last_built_value = str(last_built_at) if last_built_at else None

    target_manifest.data["frontend"] = {
        "base": _normalize_manifest_path_value(frontend_base, run_dir=run_dir_path),
        "dir": _normalize_manifest_path_value(review_dir_path, run_dir=run_dir_path),
        "packs": _normalize_manifest_path_value(packs_dir_path, run_dir=run_dir_path),
        "packs_dir": _normalize_manifest_path_value(packs_dir_path, run_dir=run_dir_path),
        "results": _normalize_manifest_path_value(responses_dir_path, run_dir=run_dir_path),
        "results_dir": _normalize_manifest_path_value(
            responses_dir_path, run_dir=run_dir_path
        ),
        "index": _normalize_manifest_path_value(index_path, run_dir=run_dir_path),
        "legacy_index": _normalize_manifest_path_value(
            legacy_index_path, run_dir=run_dir_path
        ),
        "built": bool(built),
        "packs_count": packs_count_value,
        "counts": {
            "packs": packs_count_value,
            "responses": responses_count,
        },
        "last_built_at": last_built_value,
        "last_responses_at": now_iso,
    }

    persist_manifest(target_manifest)
    return target_manifest


def _ensure_stage_status_payload(manifest: RunManifest, stage_key: str) -> dict:
    data = manifest.data
    if not isinstance(data, dict):
        data = {}
        manifest.data = data

    ai_section = data.setdefault("ai", {})
    if not isinstance(ai_section, dict):
        ai_section = {}
        data["ai"] = ai_section

    packs_section = ai_section.setdefault("packs", {})
    if not isinstance(packs_section, dict):
        packs_section = {}
        ai_section["packs"] = packs_section

    stage_section = packs_section.setdefault(stage_key, {})
    if not isinstance(stage_section, dict):
        stage_section = {}
        packs_section[stage_key] = stage_section

    status_payload = stage_section.setdefault("status", {})
    if not isinstance(status_payload, dict):
        status_payload = {}
        stage_section["status"] = status_payload

    status_payload.setdefault("built", False)
    status_payload.setdefault("sent", False)
    status_payload.setdefault("completed_at", None)

    return status_payload


def update_manifest_ai_stage_result(
    sid: str,
    stage: str,
    *,
    manifest: Optional[RunManifest] = None,
    runs_root: Optional[Path | str] = None,
    completed_at: Optional[str] = None,
) -> RunManifest:
    """Mark ``stage`` as sent with ``completed_at`` inside the manifest."""

    stage_key = str(stage).strip().lower()
    if not stage_key:
        raise ValueError("stage is required")

    target_manifest = _resolve_manifest(
        sid, manifest=manifest, runs_root=runs_root
    )

    status_payload = _ensure_stage_status_payload(target_manifest, stage_key)
    stage_status = target_manifest.ensure_ai_stage_status(stage_key)

    existing_completed = status_payload.get("completed_at")
    if isinstance(existing_completed, str) and existing_completed.strip():
        timestamp = existing_completed.strip()
    else:
        timestamp_candidate = str(completed_at).strip() if completed_at else ""
        timestamp = timestamp_candidate or _now_iso()

    changed = False

    if not bool(status_payload.get("sent")):
        status_payload["sent"] = True
        changed = True
    elif status_payload.get("sent") is not True:
        status_payload["sent"] = True
        changed = True

    if status_payload.get("completed_at") != timestamp:
        status_payload["completed_at"] = timestamp
        changed = True

    if not bool(stage_status.get("sent")):
        stage_status["sent"] = True
        changed = True
    elif stage_status.get("sent") is not True:
        stage_status["sent"] = True
        changed = True

    if stage_status.get("completed_at") != timestamp:
        stage_status["completed_at"] = timestamp
        changed = True

    if changed:
        persist_manifest(target_manifest)

    return target_manifest


__all__ = [
    "update_manifest_state",
    "update_manifest_frontend",
    "update_manifest_ai_stage_result",
]

