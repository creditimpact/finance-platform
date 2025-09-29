"""Path helpers for AI merge packs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class MergePaths:
    """Resolved filesystem locations for merge AI packs."""

    base: Path
    packs_dir: Path
    results_dir: Path
    log_file: Path
    index_file: Path


def ensure_merge_paths(runs_root: Path, sid: str, create: bool = True) -> MergePaths:
    """Return the canonical merge AI pack paths for ``sid``.

    When ``create`` is ``True`` (the default) the base directory along with the
    ``packs`` and ``results`` subdirectories are created if they do not already
    exist. When ``create`` is ``False`` the paths are computed without touching
    the filesystem.
    """

    base = Path(runs_root) / sid / "ai_packs" / "merge"
    packs_dir = base / "packs"
    results_dir = base / "results"

    if create:
        packs_dir.mkdir(parents=True, exist_ok=True)
        results_dir.mkdir(parents=True, exist_ok=True)

    return MergePaths(
        base=base,
        packs_dir=packs_dir,
        results_dir=results_dir,
        log_file=base / "logs.txt",
        index_file=base / "index.json",
    )


def pair_pack_filename(a_idx: int, b_idx: int) -> str:
    """Return the canonical filename for a pair pack."""

    lo, hi = sorted((a_idx, b_idx))
    return f"pair_{lo:03d}_{hi:03d}.jsonl"


def pair_result_filename(a_idx: int, b_idx: int) -> str:
    """Return the canonical filename for a pair result."""

    lo, hi = sorted((a_idx, b_idx))
    return f"pair_{lo:03d}_{hi:03d}.result.json"


def get_merge_paths(runs_root: Path, sid: str, *, create: bool = True) -> MergePaths:
    """Return the resolved merge AI pack paths for ``sid``."""

    return ensure_merge_paths(runs_root, sid, create=create)


def probe_legacy_ai_packs(runs_root: Path, sid: str) -> Optional[Path]:
    """Return the legacy ``ai_packs`` directory if it contains any pair packs."""

    legacy_dir = Path(runs_root) / sid / "ai_packs"
    if not legacy_dir.is_dir():
        return None

    if any(legacy_dir.glob("pair_*.jsonl")):
        return legacy_dir

    return None

