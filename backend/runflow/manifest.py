"""Helpers for synchronizing runflow decisions with the run manifest."""

from __future__ import annotations

import glob
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from backend.core.paths.frontend_review import ensure_frontend_review_dirs
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

    run_dir = target_manifest.path.parent
    canonical_paths = ensure_frontend_review_dirs(str(run_dir))

    packs_dir_path = canonical_paths["packs_dir"]
    responses_dir_path = canonical_paths["responses_dir"]
    review_dir_path = canonical_paths["review_dir"]
    frontend_base = canonical_paths["frontend_base"]
    index_path = canonical_paths["index"]

    packs_count_glob = len(glob.glob(os.path.join(packs_dir_path, "idx-*.json")))
    packs_count_param = int(packs_count or 0)
    packs_count_value = max(packs_count_glob, packs_count_param)

    responses_count = len(glob.glob(os.path.join(responses_dir_path, "*.json")))
    now_iso = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    last_built_value: str | None
    if built:
        last_built_value = (
            str(last_built_at) if last_built_at else now_iso
        )
    else:
        last_built_value = str(last_built_at) if last_built_at else None

    target_manifest.data["frontend"] = {
        "base": frontend_base,
        "dir": review_dir_path,
        "packs": packs_dir_path,
        "packs_dir": packs_dir_path,
        "results": responses_dir_path,
        "results_dir": responses_dir_path,
        "index": index_path,
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


__all__ = ["update_manifest_state", "update_manifest_frontend"]

