"""Helpers for working with run-level AI manifest documents."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


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


__all__ = ["StageManifestPaths", "extract_stage_manifest_paths"]
