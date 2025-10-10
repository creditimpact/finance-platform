"""Helpers for synchronizing runflow decisions with the run manifest."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from backend.pipeline.runs import RunManifest, persist_manifest


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

    target_manifest: RunManifest
    if manifest is not None:
        target_manifest = manifest
    else:
        if runs_root is not None:
            base = Path(runs_root)
            manifest_path = base / sid / "manifest.json"
            target_manifest = RunManifest.load_or_create(manifest_path, sid=sid)
        else:
            target_manifest = RunManifest.for_sid(sid)

    target_manifest.data["status"] = str(state)
    persist_manifest(target_manifest)
    return target_manifest


__all__ = ["update_manifest_state"]

