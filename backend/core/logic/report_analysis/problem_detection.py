from __future__ import annotations


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

    }

    if ENABLE_AI_ADJUDICATOR:
        redacted = redact_account_for_ai(acct)
        ai_resp = ai_adjudicate("stageA", "v1", redacted)
        result["debug"]["ai_response"] = {
            "primary_issue": ai_resp.get("primary_issue"),
            "confidence": ai_resp.get("confidence"),
            "tier": ai_resp.get("tier"),
            "error": ai_resp.get("error"),
        }
        if (
            ai_resp.get("confidence", 0) >= AI_MIN_CONFIDENCE
            and ai_resp.get("primary_issue") != "unknown"
        ):
            result.update(
                {
                    "primary_issue": ai_resp.get("primary_issue", "unknown"),
                    "issue_types": ai_resp.get("issue_types", []),
                    "problem_reasons": ai_resp.get("problem_reasons", []),
                    "confidence": ai_resp.get("confidence", 0.0),
                    "tier": ai_resp.get("tier", 0),
                    "decision_source": "ai",
                }
            )

    return result
