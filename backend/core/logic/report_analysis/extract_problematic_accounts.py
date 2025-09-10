from __future__ import annotations

from typing import Any, Dict, List

from backend.core.case_store.storage import CaseStoreError, load_session_case
from backend.core.orchestrators import (
    collect_stageA_problem_accounts as get_problem_accounts_for_session,
)

from .problem_case_builder import build_problem_cases


def extract_problematic_accounts(session_id: str) -> List[Dict[str, Any]]:
    """Return problematic account summaries for ``session_id``.

    If a legacy Case Store file exists we reuse the Stage‑A orchestrator results
    and normalise them into the legacy structure.  When that file is absent we
    fall back to :func:`build_problem_cases` which writes new problem case
    artifacts and returns their summaries.  In both situations a list of
    dictionaries is returned and the function never raises when no accounts are
    problematic.
    """

    try:
        load_session_case(session_id)
    except CaseStoreError:
        summary = build_problem_cases(session_id)
        if isinstance(summary, dict):
            return list(summary.get("summaries") or [])
        return []

    rows = get_problem_accounts_for_session(session_id)
    out: List[Dict[str, Any]] = []
    for r in rows:
        out.append(
            {
                "account_id": r["account_id"],
                "bureau": r["bureau"],
                "primary_issue": r["primary_issue"],
                "tier": r["tier"],
                "problem_reasons": r.get("problem_reasons", []),
                "decision_source": r.get("decision_source", "rules"),
                "confidence": float(r.get("confidence", 0.0)),
                "fields_used": r.get("fields_used", []),
            }
        )
    return out
