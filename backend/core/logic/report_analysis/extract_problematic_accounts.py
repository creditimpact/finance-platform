from __future__ import annotations

from typing import Any, Dict, List

from backend.core.orchestrators import collect_stageA_problem_accounts as get_problem_accounts_for_session


def extract_problematic_accounts(session_id: str) -> List[Dict[str, Any]]:
    """Return legacy problem account records for ``session_id``.

    This adapter bridges older callers that expect a simple list of dicts with the
    new Case Store + Stage A orchestration pipeline.  Filtering logic and schema
    validation live in :mod:`backend.core.orchestrators`; this function simply
    reads those results and normalises field names and defaults so the legacy
    contract remains stable.
    """

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
