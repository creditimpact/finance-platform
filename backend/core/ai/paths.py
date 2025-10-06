# NOTE: keep docstring to describe module responsibilities
"""Path helpers for AI adjudication artifacts."""

from __future__ import annotations

import os

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class ValidationAccountPaths:
    """Resolved filesystem locations for a single validation AI pack."""

    account_id: int
    pack_file: Path
    prompt_file: Path
    result_jsonl_file: Path
    result_summary_file: Path


@dataclass(frozen=True)
class ValidationPaths:
    """Resolved filesystem locations for validation AI packs."""

    base: Path
    packs_dir: Path
    results_dir: Path
    index_file: Path
    log_file: Path


def ensure_validation_paths(
    runs_root: Path, sid: str, create: bool = True
) -> ValidationPaths:
    """Return the canonical validation AI pack paths for ``sid``."""

    runs_root_path = Path(runs_root).resolve()
    base_path = (runs_root_path / sid / "ai_packs" / "validation").resolve()

    def _resolve_override(env_name: str, default: Path) -> Path:
        raw = os.getenv(env_name)
        if not raw:
            return default

        candidate = Path(raw)
        if not candidate.is_absolute():
            candidate = (runs_root_path / sid / candidate).resolve()
        else:
            candidate = candidate.resolve()
        return candidate

    packs_dir = _resolve_override("VALIDATION_PACKS_DIR", base_path / "packs")
    results_dir = _resolve_override("VALIDATION_RESULTS_DIR", base_path / "results")
    index_file = base_path / "index.json"
    log_file = base_path / "logs.txt"

    if create:
        base_path.mkdir(parents=True, exist_ok=True)
        packs_dir.mkdir(parents=True, exist_ok=True)
        results_dir.mkdir(parents=True, exist_ok=True)

    return ValidationPaths(
        base=base_path,
        packs_dir=packs_dir.resolve(),
        results_dir=results_dir.resolve(),
        index_file=index_file.resolve(strict=False),
        log_file=log_file.resolve(strict=False),
    )


def ensure_validation_account_paths(
    paths: ValidationPaths, account_idx: int | str, *, create: bool = True
) -> ValidationAccountPaths:
    """Return filesystem locations for ``account_idx`` under ``paths``."""

    try:
        normalized_idx = int(str(account_idx))
    except (TypeError, ValueError):
        raise ValueError("account_idx must be coercible to an integer") from None

    pack_filename = validation_pack_filename_for_account(normalized_idx)
    pack_file = paths.packs_dir / pack_filename
    prompt_file = paths.packs_dir / f"{pack_filename}.prompt.txt"
    result_jsonl_file = (
        paths.results_dir / validation_result_jsonl_filename_for_account(normalized_idx)
    )
    result_summary_file = (
        paths.results_dir / validation_result_filename_for_account(normalized_idx)
    )

    if create:
        pack_file.parent.mkdir(parents=True, exist_ok=True)
        prompt_file.parent.mkdir(parents=True, exist_ok=True)
        result_jsonl_file.parent.mkdir(parents=True, exist_ok=True)
        result_summary_file.parent.mkdir(parents=True, exist_ok=True)

    return ValidationAccountPaths(
        account_id=normalized_idx,
        pack_file=pack_file,
        prompt_file=prompt_file,
        result_jsonl_file=result_jsonl_file,
        result_summary_file=result_summary_file,
    )


@dataclass(frozen=True)
class MergePaths:
    """Resolved filesystem locations for merge AI packs."""

    base: Path
    packs_dir: Path
    results_dir: Path
    log_file: Path
    index_file: Path


def _merge_paths_from_base(base: Path, *, create: bool) -> MergePaths:
    base_path = Path(base).resolve()
    packs_dir = base_path / "packs"
    results_dir = base_path / "results"

    if create:
        packs_dir.mkdir(parents=True, exist_ok=True)
        results_dir.mkdir(parents=True, exist_ok=True)

    return MergePaths(
        base=base_path,
        packs_dir=packs_dir,
        results_dir=results_dir,
        log_file=base_path / "logs.txt",
        index_file=base_path / "index.json",
    )


def ensure_merge_paths(runs_root: Path, sid: str, create: bool = True) -> MergePaths:
    """Return the canonical merge AI pack paths for ``sid``.

    When ``create`` is ``True`` (the default) the base directory along with the
    ``packs`` and ``results`` subdirectories are created if they do not already
    exist. When ``create`` is ``False`` the paths are computed without touching
    the filesystem.
    """

    base = Path(runs_root) / sid / "ai_packs" / "merge"
    return _merge_paths_from_base(base, create=create)


def merge_paths_from_any(path: Path | str, *, create: bool = False) -> MergePaths:
    """Return :class:`MergePaths` using ``path`` rooted at the merge base.

    ``path`` may point at the merge base itself (``.../merge``) or one of its
    canonical children (``.../merge/packs`` or ``.../merge/results``).  The
    caller controls directory creation via ``create``; by default this function
    is read-only.
    """

    resolved = Path(path).resolve()
    if resolved.name == "merge":
        return _merge_paths_from_base(resolved, create=create)
    if resolved.parent.name == "merge" and resolved.name in {"packs", "results"}:
        return _merge_paths_from_base(resolved.parent, create=create)

    raise ValueError(f"Path does not identify merge layout: {resolved}")


def pair_pack_filename(a_idx: int, b_idx: int) -> str:
    """Return the canonical filename for a pair pack."""

    lo, hi = sorted((a_idx, b_idx))
    return f"pair_{lo:03d}_{hi:03d}.jsonl"


def pair_result_filename(a_idx: int, b_idx: int) -> str:
    """Return the canonical filename for a pair result."""

    lo, hi = sorted((a_idx, b_idx))
    return f"pair_{lo:03d}_{hi:03d}.result.json"


def pair_pack_path(paths: MergePaths, a_idx: int, b_idx: int) -> Path:
    """Return the resolved filesystem path for a pair pack."""

    return paths.packs_dir / pair_pack_filename(a_idx, b_idx)


def pair_result_path(paths: MergePaths, a_idx: int, b_idx: int) -> Path:
    """Return the resolved filesystem path for a pair result."""

    return paths.results_dir / pair_result_filename(a_idx, b_idx)


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


def _resolve_runs_root(runs_root: Path | str | None) -> Path:
    """Return the effective runs root using ``RUNS_ROOT`` env fallback."""

    if runs_root is None:
        env_root = os.getenv("RUNS_ROOT")
        return Path(env_root) if env_root else Path("runs")
    return Path(runs_root)


def validation_base_dir(
    sid: str, runs_root: Path | str | None = None, *, create: bool = True
) -> Path:
    """Return the canonical validation base directory for ``sid``.

    When ``create`` is ``True`` the directory is created if it does not exist.
    """

    base = _resolve_runs_root(runs_root) / sid / "ai_packs" / "validation"
    if create:
        base.mkdir(parents=True, exist_ok=True)
    return base.resolve()


def validation_packs_dir(
    sid: str, runs_root: Path | str | None = None, *, create: bool = True
) -> Path:
    """Return the directory holding validation pack payloads for ``sid``."""

    packs_dir = validation_base_dir(sid, runs_root=runs_root, create=create) / "packs"
    if create:
        packs_dir.mkdir(parents=True, exist_ok=True)
    return packs_dir.resolve()


def validation_results_dir(
    sid: str, runs_root: Path | str | None = None, *, create: bool = True
) -> Path:
    """Return the directory holding validation model results for ``sid``."""

    results_dir = (
        validation_base_dir(sid, runs_root=runs_root, create=create) / "results"
    )
    if create:
        results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir.resolve()


def validation_index_path(
    sid: str, runs_root: Path | str | None = None, *, create: bool = True
) -> Path:
    """Return the manifest index path for validation packs."""

    runs_root_path = _resolve_runs_root(runs_root).resolve()
    validation_paths = ensure_validation_paths(runs_root_path, sid, create=create)
    return validation_paths.index_file


def validation_logs_path(
    sid: str, runs_root: Path | str | None = None, *, create: bool = True
) -> Path:
    """Return the log file path for validation pack activity."""

    base = validation_base_dir(sid, runs_root=runs_root, create=create)
    if create:
        base.mkdir(parents=True, exist_ok=True)
    return (base / "logs.txt").resolve()


def _normalize_account_id(account_id: int | str) -> int:
    try:
        return int(str(account_id).strip())
    except (TypeError, ValueError):  # pragma: no cover - defensive
        raise ValueError("account_id must be coercible to an integer") from None


def validation_pack_filename_for_account(account_id: int | str) -> str:
    """Return the canonical validation pack filename for ``account_id``."""

    normalized = _normalize_account_id(account_id)
    return f"val_acc_{normalized:03d}.jsonl"


def validation_result_jsonl_filename_for_account(account_id: int | str) -> str:
    """Return the canonical validation result JSONL filename for ``account_id``."""

    normalized = _normalize_account_id(account_id)
    return f"acc_{normalized:03d}.result.jsonl"


def validation_result_summary_filename_for_account(account_id: int | str) -> str:
    """Return the canonical validation result summary filename for ``account_id``."""

    normalized = _normalize_account_id(account_id)
    return f"acc_{normalized:03d}.result.json"


def validation_result_filename_for_account(account_id: int | str) -> str:
    """Backward-compatible alias for the summary filename."""

    return validation_result_summary_filename_for_account(account_id)

