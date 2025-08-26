from __future__ import annotations

from typing import Any, Dict, List


def evaluate_account_problem(acct: Dict[str, Any]) -> Dict[str, Any]:
    """Detect potential problems on a credit account without classification.

    The function only surfaces accounts that show evidence of issues such as
    late payments or past-due amounts. It does not attempt to categorize the
    account or assign issue types. All accounts flagged here will therefore
    carry a neutral ``primary_issue`` of ``"unknown"``.
    """

    reasons: List[str] = []
    supporting: Dict[str, Any] = {}

    # Late payment history
    late = acct.get("late_payments") or {}
    for bureau, buckets in late.items():
        for days, count in (buckets or {}).items():
            try:
                c = int(count)
                d = int(days)
            except Exception:
                continue
            if c > 0:
                reasons.append(f"late_payment: {c}\u00d7{d} on {bureau}")
    if late:
        supporting["late_payments"] = late

    # Past due amount
    if acct.get("past_due_amount") is not None:
        supporting["past_due_amount"] = acct["past_due_amount"]
        try:
            if float(acct["past_due_amount"]) > 0:
                reasons.append("past_due_amount")
        except Exception:
            pass

    is_problem = bool(reasons)

    return {
        "is_problem": is_problem,
        "primary_issue": "unknown",
        "problem_reasons": reasons,
        "decision_source": "rules",
        "confidence": 0.0,
        "supporting": supporting,
    }
