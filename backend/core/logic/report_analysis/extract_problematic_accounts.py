"""Orchestrate extraction and case building for problematic accounts.

This module provides a thin wrapper that wires together the problem
extraction and case building steps.  It intentionally contains *no*
problem detection logic or file writing beyond delegating to the
specialised modules.  This keeps responsibilities clear:

``problem_extractor``  -> decides which accounts are problematic
``problem_case_builder`` -> persists per-account case files
``extract_problematic_accounts`` -> orchestrates the two steps
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List

from .account_merge import DEFAULT_CFG, cluster_problematic_accounts
from .problem_case_builder import build_problem_cases
from .problem_extractor import detect_problem_accounts


def extract_problematic_accounts(
    session_id: str, *, root: Path | None = None
) -> Dict[str, Any]:
    """Run problem extraction then build case artifacts.

    Parameters
    ----------
    session_id:
        Identifier of the current session whose accounts should be
        analysed.
    root:
        Optional repository root used for locating artefacts.  When not
        supplied the defaults from the underlying modules are used.

    Returns
    -------
    dict
        Dictionary containing the list of problematic accounts under the
        ``found`` key and the builder summary under ``summary``.
    """

    candidates: List[Dict[str, Any]] = detect_problem_accounts(session_id, root=root)
    summary = build_problem_cases(session_id, candidates=candidates, root=root)

    if root is not None:
        default_runs_root = Path(root)
    else:
        default_runs_root = Path("runs")

    runs_root = Path(os.environ.get("RUNS_ROOT", str(default_runs_root)))

    merged_candidates = cluster_problematic_accounts(
        candidates,
        DEFAULT_CFG,
        sid=session_id,
        runs_root=runs_root,
    )

    return {"found": merged_candidates, "summary": summary}


__all__ = ["extract_problematic_accounts"]

