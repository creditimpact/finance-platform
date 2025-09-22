"""Wrappers around merge scoring helpers for programmatic use."""

from __future__ import annotations

import os
from pathlib import Path

from scripts.score_bureau_pairs import ScoreComputationResult, score_accounts

_DEFAULT_RUNS_ROOT = Path(os.environ.get("RUNS_ROOT", "runs"))


def score_bureau_pairs_cli(
    *, sid: str, write_tags: bool = False, runs_root: Path | str | None = None
) -> ScoreComputationResult:
    """Execute the score-bureau-pairs workflow similar to the CLI.

    Parameters
    ----------
    sid:
        The run/session identifier.
    write_tags:
        When ``True`` persists merge tags back to account ``tags.json`` files.
    runs_root:
        Optional override for the runs directory root.
    """

    sid_str = str(sid)
    runs_root_path = Path(runs_root) if runs_root is not None else _DEFAULT_RUNS_ROOT
    return score_accounts(sid_str, runs_root=runs_root_path, write_tags=write_tags)


__all__ = ["score_bureau_pairs_cli"]
