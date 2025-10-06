"""Helpers for working with run-level AI manifest documents."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from backend.core.ai.paths import ensure_validation_paths
from backend.pipeline.runs import RunManifest, persist_manifest


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


__all__ = ["Manifest", "StageManifestPaths", "extract_stage_manifest_paths"]
