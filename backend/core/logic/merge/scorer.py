"""Wrappers around merge scoring helpers for programmatic use."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from backend import config as app_config
from scripts.score_bureau_pairs import ScoreComputationResult, score_accounts

_DEFAULT_RUNS_ROOT = Path(os.environ.get("RUNS_ROOT", "runs"))

logger = logging.getLogger(__name__)

SCORER_WEIGHTS = {
    "acctnum_exact": app_config.ACCTNUM_EXACT_WEIGHT,
    "last5": app_config.ACCTNUM_LAST5_WEIGHT,
    "last4": app_config.ACCTNUM_LAST4_WEIGHT,
    "masked": app_config.ACCTNUM_MASKED_WEIGHT,
}


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
    logger.info(
        "SCORER_WEIGHTS acctnum_exact=%s last5=%s last4=%s masked=%s",
        SCORER_WEIGHTS["acctnum_exact"],
        SCORER_WEIGHTS["last5"],
        SCORER_WEIGHTS["last4"],
        SCORER_WEIGHTS["masked"],
    )
    runs_root_path = Path(runs_root) if runs_root is not None else _DEFAULT_RUNS_ROOT
    return score_accounts(sid_str, runs_root=runs_root_path, write_tags=write_tags)


__all__ = ["score_bureau_pairs_cli"]
