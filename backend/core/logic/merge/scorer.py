"""Wrappers around merge scoring helpers for programmatic use."""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict

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


def _ensure_tag_levels(
    merge_tags: Mapping[int, Mapping[str, Any]] | None,
) -> Dict[int, Dict[str, Any]]:
    enriched: Dict[int, Dict[str, Any]] = {}
    if not isinstance(merge_tags, Mapping):
        return enriched

    for idx, tag in merge_tags.items():
        if not isinstance(tag, Mapping):
            continue
        tag_dict = dict(tag)
        acct_level = tag_dict.get("acctnum_level")
        if not isinstance(acct_level, str) or not acct_level:
            aux_payload = tag_dict.get("aux")
            if isinstance(aux_payload, Mapping):
                acct_level = str(aux_payload.get("acctnum_level") or "none")
            else:
                acct_level = "none"
        tag_dict["acctnum_level"] = acct_level
        try:
            key = int(idx)
        except (TypeError, ValueError):
            continue
        enriched[key] = tag_dict

    return enriched


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
    computation = score_accounts(
        sid_str, runs_root=runs_root_path, write_tags=write_tags
    )
    enriched_tags = _ensure_tag_levels(computation.merge_tags)
    if enriched_tags:
        computation = replace(computation, merge_tags=enriched_tags)
    return computation


__all__ = ["score_bureau_pairs_cli"]
