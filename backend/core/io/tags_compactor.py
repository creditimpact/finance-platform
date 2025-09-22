"""Helpers to compact tags for entire runs."""

from __future__ import annotations

import os
from pathlib import Path

from .tags_minimize import compact_account_tags

_DEFAULT_RUNS_ROOT = Path(os.environ.get("RUNS_ROOT", "runs"))


def compact_tags_for_sid(sid: str, runs_root: Path | str | None = None) -> None:
    """Compact ``tags.json`` files and summaries for all accounts of ``sid``."""

    root = Path(runs_root) if runs_root is not None else _DEFAULT_RUNS_ROOT
    accounts_root = root / str(sid) / "cases" / "accounts"
    if not accounts_root.exists():
        return

    for entry in sorted(accounts_root.iterdir()):
        if entry.is_dir():
            compact_account_tags(entry)


__all__ = ["compact_tags_for_sid"]
