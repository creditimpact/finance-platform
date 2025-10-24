"""Helpers for working with run-level AI manifest documents."""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from backend.core.ai.paths import ensure_note_style_paths, ensure_validation_paths
from backend.pipeline.runs import RUNS_ROOT_ENV, RunManifest, persist_manifest


@dataclass(frozen=True)
class StageManifestPaths:
    """Resolved filesystem paths for a specific AI stage."""

    base_dir: Path | None = None
    packs_dir: Path | None = None
    results_dir: Path | None = None
    index_file: Path | None = None
    log_file: Path | None = None

    def has_any(self) -> bool:
        """Return ``True`` when at least one path is populated."""

        return any(
            value is not None
            for value in (
                self.base_dir,
                self.packs_dir,
                self.results_dir,
                self.index_file,
                self.log_file,
            )
        )


def _coerce_path(value: Any) -> Path | None:
    if value is None:
        return None

    try:
        text = os.fspath(value)
    except TypeError:
        return None

    stripped = str(text).strip()
    if not stripped:
        return None

    try:
        return Path(stripped).resolve()
    except OSError:
        return Path(stripped)


def extract_stage_manifest_paths(
    manifest: Mapping[str, Any], stage: str
) -> StageManifestPaths:
    """Return the preferred filesystem paths for ``stage`` within ``manifest``."""

    stage_key = stage.lower().strip()

    ai_section = manifest.get("ai")
    if not isinstance(ai_section, Mapping):
        return StageManifestPaths()

    base_dir: Path | None = None
    packs_dir: Path | None = None
    results_dir: Path | None = None
    index_file: Path | None = None
    log_file: Path | None = None

    packs_section = ai_section.get("packs")
    if isinstance(packs_section, Mapping):
        stage_section = packs_section.get(stage_key)
        if isinstance(stage_section, Mapping):
            base_dir = _coerce_path(stage_section.get("base")) or _coerce_path(
                stage_section.get("dir")
            )
            packs_dir = _coerce_path(stage_section.get("packs_dir")) or _coerce_path(
                stage_section.get("packs")
            )
            results_dir = _coerce_path(stage_section.get("results_dir")) or _coerce_path(
                stage_section.get("results")
            )
            index_file = _coerce_path(stage_section.get("index"))
            log_file = _coerce_path(stage_section.get("logs"))

    legacy_stage = ai_section.get(stage_key)
    if isinstance(legacy_stage, Mapping):
        legacy_base = (
            _coerce_path(legacy_stage.get("dir"))
            or _coerce_path(legacy_stage.get("base"))
            or _coerce_path(legacy_stage.get("accounts_dir"))
            or _coerce_path(legacy_stage.get("accounts"))
        )
        if legacy_base is not None:
            if base_dir is None:
                base_dir = legacy_base
            if packs_dir is None:
                packs_dir = (legacy_base / "packs").resolve()
            if results_dir is None:
                results_dir = (legacy_base / "results").resolve()
            if index_file is None:
                index_file = (legacy_base / "index.json").resolve()
            if log_file is None:
                log_file = (legacy_base / "logs.txt").resolve()

    return StageManifestPaths(
        base_dir=base_dir,
        packs_dir=packs_dir,
        results_dir=results_dir,
        index_file=index_file,
        log_file=log_file,
    )


log = logging.getLogger(__name__)


def _now_iso_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


class Manifest:
    """Helpers for mutating run-level manifest documents."""

    @staticmethod
    def ensure_validation_section(
        sid: str, *, runs_root: Path | str | None = None
    ) -> dict[str, Any]:
        """Ensure the validation packs section exists for ``sid``.

        The manifest is written to disk when any values are injected.  The
        function always creates the canonical validation directories so the
        manifest can reference them immediately.
        """

        sid_text = str(sid).strip()
        if not sid_text:
            raise ValueError("sid is required")

        runs_root_path: Path | None
        if runs_root is not None:
            runs_root_path = Path(runs_root).resolve()
            manifest_path = runs_root_path / sid_text / "manifest.json"
            manifest = RunManifest.load_or_create(manifest_path, sid_text)
        else:
            manifest = RunManifest.for_sid(sid_text)
            runs_root_path = manifest.path.parent.parent.resolve()

        validation_paths = ensure_validation_paths(
            runs_root_path, sid_text, create=True
        )

        data = manifest.data
        if not isinstance(data, dict):
            data = {}
            manifest.data = data

        ai_section = data.get("ai")
        if not isinstance(ai_section, dict):
            ai_section = {}
            data["ai"] = ai_section

        packs_section = ai_section.get("packs")
        if not isinstance(packs_section, dict):
            packs_section = {}
            ai_section["packs"] = packs_section

        validation_section = packs_section.get("validation")
        if not isinstance(validation_section, dict):
            validation_section = {}
            packs_section["validation"] = validation_section

        canonical_values = {
            "base": str(validation_paths.base),
            "dir": str(validation_paths.base),
            "packs": str(validation_paths.packs_dir),
            "packs_dir": str(validation_paths.packs_dir),
            "results": str(validation_paths.results_dir),
            "results_dir": str(validation_paths.results_dir),
            "index": str(validation_paths.index_file),
            "logs": str(validation_paths.log_file),
        }

        changed = False
        for key, value in canonical_values.items():
            current = validation_section.get(key)
            if not isinstance(current, str) or not current.strip():
                validation_section[key] = value
                changed = True

        if changed:
            persist_manifest(manifest)

        packs_dir = validation_section.get("packs_dir") or canonical_values["packs_dir"]
        results_dir = (
            validation_section.get("results_dir") or canonical_values["results_dir"]
        )

        log.info(
            "VALIDATION_MANIFEST_INJECTED sid=%s packs_dir=%s results_dir=%s",
            sid_text,
            packs_dir,
            results_dir,
        )

        return dict(validation_section)

    @staticmethod
    def ensure_note_style_section(
        sid: str, *, runs_root: Path | str | None = None
    ) -> dict[str, Any]:
        """Ensure the note_style packs section exists for ``sid``."""

        sid_text = str(sid).strip()
        if not sid_text:
            raise ValueError("sid is required")

        runs_root_path: Path | None
        manifest: RunManifest
        previous_runs_root = None
        try:
            if runs_root is not None:
                runs_root_path = Path(runs_root).resolve()
                manifest_path = runs_root_path / sid_text / "manifest.json"
                previous_runs_root = os.getenv(RUNS_ROOT_ENV)
                os.environ[RUNS_ROOT_ENV] = str(runs_root_path)
                manifest = RunManifest.load_or_create(manifest_path, sid_text)
            else:
                manifest = RunManifest.for_sid(sid_text)
                runs_root_path = manifest.path.parent.parent.resolve()
        finally:
            if runs_root is not None:
                if previous_runs_root is None:
                    os.environ.pop(RUNS_ROOT_ENV, None)
                else:
                    os.environ[RUNS_ROOT_ENV] = previous_runs_root

        note_style_paths = ensure_note_style_paths(
            runs_root_path, sid_text, create=True
        )

        data = manifest.data
        if not isinstance(data, dict):
            data = {}
            manifest.data = data

        ai_section = data.get("ai")
        if not isinstance(ai_section, dict):
            ai_section = {}
            data["ai"] = ai_section

        packs_section = ai_section.get("packs")
        if not isinstance(packs_section, dict):
            packs_section = {}
            ai_section["packs"] = packs_section

        note_style_section = packs_section.get("note_style")
        if not isinstance(note_style_section, dict):
            note_style_section = {}
            packs_section["note_style"] = note_style_section

        canonical_values = {
            "base": str(note_style_paths.base),
            "dir": str(note_style_paths.base),
            "packs": str(note_style_paths.packs_dir),
            "packs_dir": str(note_style_paths.packs_dir),
            "results": str(note_style_paths.results_dir),
            "results_dir": str(note_style_paths.results_dir),
            "index": str(note_style_paths.index_file),
            "logs": str(note_style_paths.base / "logs.txt"),
        }

        changed = False
        for key, value in canonical_values.items():
            current = note_style_section.get(key)
            if not isinstance(current, str) or not current.strip():
                note_style_section[key] = value
                changed = True

        if "last_built_at" not in note_style_section:
            note_style_section["last_built_at"] = None
            changed = True

        status_payload = note_style_section.get("status")
        if not isinstance(status_payload, Mapping):
            note_style_section["status"] = {"built": False, "completed_at": None}
            changed = True
        else:
            built_changed = False
            if "built" not in status_payload:
                status_payload["built"] = False
                built_changed = True
            if "completed_at" not in status_payload:
                status_payload["completed_at"] = None
                built_changed = True
            changed = changed or built_changed

        if changed:
            persist_manifest(manifest)

        packs_dir = note_style_section.get("packs_dir") or canonical_values["packs_dir"]
        results_dir = (
            note_style_section.get("results_dir") or canonical_values["results_dir"]
        )

        log.info(
            "NOTE_STYLE_MANIFEST_INJECTED sid=%s packs_dir=%s results_dir=%s",
            sid_text,
            packs_dir,
            results_dir,
        )

        return dict(note_style_section)

    @staticmethod
    def register_note_style_build(
        sid: str,
        *,
        runs_root: Path | str | None = None,
        timestamp: str | None = None,
    ) -> dict[str, Any]:
        """Record a successful note_style build inside ``manifest.json``."""

        sid_text = str(sid).strip()
        if not sid_text:
            raise ValueError("sid is required")

        timestamp_text = str(timestamp).strip() if timestamp else ""
        if not timestamp_text:
            timestamp_text = _now_iso_utc()

        runs_root_path: Path | None
        manifest: RunManifest
        previous_runs_root = None
        try:
            if runs_root is not None:
                runs_root_path = Path(runs_root).resolve()
                manifest_path = runs_root_path / sid_text / "manifest.json"
                previous_runs_root = os.getenv(RUNS_ROOT_ENV)
                os.environ[RUNS_ROOT_ENV] = str(runs_root_path)
                manifest = RunManifest.load_or_create(manifest_path, sid_text)
            else:
                manifest = RunManifest.for_sid(sid_text)
                runs_root_path = manifest.path.parent.parent.resolve()
        finally:
            if runs_root is not None:
                if previous_runs_root is None:
                    os.environ.pop(RUNS_ROOT_ENV, None)
                else:
                    os.environ[RUNS_ROOT_ENV] = previous_runs_root

        note_style_paths = ensure_note_style_paths(
            runs_root_path, sid_text, create=True
        )

        manifest.upsert_note_style_packs_dir(
            note_style_paths.base,
            packs_dir=note_style_paths.packs_dir,
            results_dir=note_style_paths.results_dir,
            index_file=note_style_paths.index_file,
            log_file=note_style_paths.log_file,
            last_built_at=timestamp_text,
        )

        data = manifest.data if isinstance(manifest.data, Mapping) else {}
        ai_section = data.get("ai") if isinstance(data, Mapping) else {}
        packs_section = ai_section.get("packs") if isinstance(ai_section, Mapping) else {}
        note_style_section: Mapping[str, Any] | None = None
        if isinstance(packs_section, Mapping):
            candidate = packs_section.get("note_style")
            if isinstance(candidate, Mapping):
                note_style_section = candidate

        log.info(
            "NOTE_STYLE_MANIFEST_BUILT sid=%s packs_dir=%s results_dir=%s",
            sid_text,
            str(note_style_paths.packs_dir),
            str(note_style_paths.results_dir),
        )

        return dict(note_style_section or {})


def ensure_note_style_section(
    sid: str, *, runs_root: Path | str | None = None
) -> dict[str, Any]:
    """Ensure the note_style manifest section and directories exist for ``sid``."""

    return Manifest.ensure_note_style_section(sid, runs_root=runs_root)


def register_note_style_build(
    sid: str,
    *,
    runs_root: Path | str | None = None,
    timestamp: str | None = None,
) -> dict[str, Any]:
    """Register note_style build completion in the run manifest."""

    return Manifest.register_note_style_build(
        sid, runs_root=runs_root, timestamp=timestamp
    )


def ensure_validation_section(
    sid: str, *, runs_root: Path | str | None = None
) -> dict[str, Any]:
    """Ensure the validation manifest section and directories exist for ``sid``."""

    return Manifest.ensure_validation_section(sid, runs_root=runs_root)


__all__ = [
    "Manifest",
    "StageManifestPaths",
    "ensure_note_style_section",
    "register_note_style_build",
    "ensure_validation_section",
    "extract_stage_manifest_paths",
]
