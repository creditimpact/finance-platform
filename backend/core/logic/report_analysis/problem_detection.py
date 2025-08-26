"""Rule/AI hybrid problem detection for Stage A."""

from __future__ import annotations

import time
from typing import Any, Dict, List

from backend.api.internal_ai import adjudicate as ai_adjudicate
from backend.config import AI_MIN_CONFIDENCE, ENABLE_AI_ADJUDICATOR

from .redaction import redact_account_for_ai


def evaluate_account_problem(acct: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate an account for potential problems.

    The function applies lightweight rule-based checks and optionally consults
    the AI adjudicator.  It returns a unified decision dictionary and mutates
    ``acct`` in-place so the decision fields are available to later stages.
    """

    reasons: List[str] = []
    late = acct.get("late_payments") or {}
    for bureau, buckets in late.items():
        for days, count in (buckets or {}).items():
            try:
                c = int(count)
                d = int(days)
            except Exception:
                continue
            if c > 0:
                reasons.append(f"late_payment: {c}x{d} on {bureau}")
    if acct.get("past_due_amount") is not None:
        try:
            if float(acct["past_due_amount"]) > 0:
                reasons.append("past_due_amount")
        except Exception:
            pass

    decision: Dict[str, Any] = {
        "primary_issue": "unknown",
        "issue_types": [],
        "problem_reasons": list(reasons),
        "tier": 0,
        "confidence": 0.0,
        "decision_source": "rules",
        "adjudicator_version": "rules-v1",
        "debug": {
            "ai_latency_ms": 0,
            "ai_tokens_in": 0,
            "ai_tokens_out": 0,
            "ai_error": None,
        },
    }

    is_problem = bool(reasons)

    if ENABLE_AI_ADJUDICATOR:
        start = time.perf_counter()
        try:
            redacted = redact_account_for_ai(acct)
            ai_resp = ai_adjudicate("stageA", "v1", redacted)
            decision["debug"].update(
                {
                    "ai_latency_ms": int((time.perf_counter() - start) * 1000),
                    "ai_tokens_in": ai_resp.get("tokens_in", 0),
                    "ai_tokens_out": ai_resp.get("tokens_out", 0),
                    "ai_error": ai_resp.get("error"),
                }
            )
            if ai_resp.get("error"):
                decision["decision_source"] = "fallback_ai_error"
            elif (
                ai_resp.get("confidence", 0.0) >= AI_MIN_CONFIDENCE
                and ai_resp.get("primary_issue") != "unknown"
            ):
                decision.update(
                    {
                        "primary_issue": ai_resp.get("primary_issue", "unknown"),
                        "issue_types": ai_resp.get("issue_types", []),
                        "problem_reasons": ai_resp.get("problem_reasons", []),
                        "confidence": ai_resp.get("confidence", 0.0),
                        "tier": ai_resp.get("tier", 0),
                        "decision_source": "ai",
                        "adjudicator_version": ai_resp.get(
                            "adjudicator_version", "ai-v1"
                        ),
                    }
                )
                is_problem = True
            else:
                decision["decision_source"] = "fallback_ai_low_conf"
        except Exception as exc:  # pragma: no cover - defensive
            decision["decision_source"] = "fallback_ai_error"
            decision["debug"]["ai_error"] = str(exc)
            decision["debug"]["ai_latency_ms"] = int(
                (time.perf_counter() - start) * 1000
            )

    acct.update({k: v for k, v in decision.items() if k != "debug"})
    acct["debug"] = decision["debug"]
    acct["_detector_is_problem"] = bool(is_problem)
    return decision
