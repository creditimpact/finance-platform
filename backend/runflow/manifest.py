"""Helpers for synchronizing runflow decisions with the run manifest."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from backend.pipeline.runs import RunManifest, persist_manifest


def _resolve_manifest(
    sid: str,
    *,
    manifest: Optional[RunManifest] = None,
    runs_root: Optional[Path | str] = None,
) -> RunManifest:
    if manifest is not None:
        return manifest

    if runs_root is not None:
        base = Path(runs_root)
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

    target_manifest.data["status"] = str(state)
    target_manifest.data["run_state"] = str(state)
    persist_manifest(target_manifest)
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

    frontend_section = target_manifest.data.get("frontend")
    if not isinstance(frontend_section, dict):
        frontend_section = {}
        target_manifest.data["frontend"] = frontend_section

    frontend_section.update(
        {
            "packs_dir": str(packs_dir) if packs_dir else None,
            "built": bool(built),
            "packs_count": int(packs_count),
            "last_built_at": str(last_built_at) if last_built_at else None,
        }
    )

    persist_manifest(target_manifest)
    return target_manifest


__all__ = ["update_manifest_state", "update_manifest_frontend"]

