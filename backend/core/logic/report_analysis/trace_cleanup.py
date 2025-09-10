from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)


_DEFAULT_ARTIFACTS: tuple[str, ...] = (
    "accounts_table/_debug_full.tsv",
    "accounts_table/accounts_from_full.json",
    "accounts_table/general_info_from_full.json",
)


def _expand_dirs(paths: Iterable[Path], base: Path) -> set[Path]:
    """Return a set of dirs that must be preserved for given paths."""
    keep_dirs: set[Path] = {base}
    for p in paths:
        for parent in p.parents:
            if parent == base:
                break
            keep_dirs.add(parent)
    return keep_dirs


def purge_trace_except_artifacts(
    sid: str,
    root: Path | str = Path("."),
    keep_extra: list[str] | None = None,
    dry_run: bool = False,
    delete_texts_sid: bool = True,
) -> dict:
    """
    Deletes everything under traces/blocks/<sid> except the 3 final artifacts listed below.
    Returns a dict summary { 'kept': [...], 'deleted': [...], 'skipped': [...], 'root': '<abs path>' }.
    Raises FileNotFoundError if the session folder doesn't exist.
    Raises RuntimeError if one of the required artifacts is missing (no deletion happens).
    """

    base = Path(root) / "traces" / "blocks" / sid
    base = base.resolve()
    if not base.exists():
        raise FileNotFoundError(base)

    keep_rel = list(_DEFAULT_ARTIFACTS)
    if keep_extra:
        keep_rel.extend(keep_extra)

    keep_abs = {base / Path(p) for p in keep_rel}

    # Verify required artifacts exist (first three of _DEFAULT_ARTIFACTS)
    for req in _DEFAULT_ARTIFACTS:
        req_path = base / req
        if not req_path.exists():
            raise RuntimeError(f"required artifact missing: {req}")

    keep_dirs = _expand_dirs(keep_abs, base)

    kept = sorted(str(p.relative_to(base)) for p in keep_abs)
    deleted: list[str] = []
    skipped: list[str] = []
    texts_deleted = False

    logger.info(
        "purge_trace_except_artifacts: sid=%s root=%s dry_run=%s", sid, base, dry_run
    )

    all_paths = sorted(base.rglob("*"), key=lambda p: len(p.parts), reverse=True)
    for path in all_paths:
        rel = str(path.relative_to(base))
        if path in keep_abs or path in keep_dirs:
            logger.debug("keeping %s", path)
            continue
        if dry_run:
            logger.debug("would delete %s", path)
            skipped.append(rel)
            continue
        try:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
            deleted.append(rel)
            logger.debug("deleted %s", path)
        except Exception:
            logger.exception("failed to delete %s", path)
            skipped.append(rel)

    if delete_texts_sid:
        texts_dir = Path(root) / "traces" / "texts" / sid
        if texts_dir.exists():
            texts_deleted = True
            rel = f"texts/{sid}/**"
            if dry_run:
                logger.debug("would delete %s", texts_dir)
                deleted.append(rel)
            else:
                try:
                    shutil.rmtree(texts_dir)
                    deleted.append(rel)
                    logger.debug("deleted %s", texts_dir)
                except Exception:
                    logger.exception("failed to delete %s", texts_dir)
                    texts_deleted = False

    logger.info(
        "purge complete: kept=%d deleted=%d skipped=%d", len(kept), len(deleted), len(skipped)
    )
    return {
        "kept": kept,
        "deleted": deleted,
        "skipped": skipped,
        "root": str(base),
        "texts_deleted": texts_deleted,
    }

def purge_after_export(sid: str, project_root: Path | str = Path(".")) -> dict:
    """Purge trace directories after export, keeping final artifacts."""
    return purge_trace_except_artifacts(sid=sid, root=Path(project_root), dry_run=False)
