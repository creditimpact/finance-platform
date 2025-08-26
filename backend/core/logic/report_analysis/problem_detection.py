import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

from backend.api.internal_ai import adjudicate as ai_adjudicate
from backend.config import (
    AI_MIN_CONFIDENCE,
    ENABLE_AI_ADJUDICATOR,
    ENABLE_TIER1_KEYWORDS,
    ENABLE_TIER2_KEYWORDS,
    ENABLE_TIER2_NUMERIC,
    ENABLE_TIER3_KEYWORDS,
    SERIOUS_DELINQUENCY_MIN_DPD,
    TIER1_KEYWORDS,
    TIER2_KEYWORDS,
    TIER3_KEYWORDS,
    UTILIZATION_PROBLEM_THRESHOLD,
)
from backend.core.logic.report_analysis.redaction import redact_account_for_ai

PRIORITY_T1 = [
    "bankruptcy",
    "foreclosure",
    "judgment",
    "tax_lien",
    "charge_off",
    "collection",
]


@dataclass
class ConfidenceHint:
    tier: int
    strongest_signal: str
    repetition_count: int
    latest_date_seen: Optional[str] = None


def _norm(v: Any) -> str:
    return (v or "").strip().lower()


def _contains_any(text: Any, needles: List[str]) -> Optional[str]:
    t = _norm(text)
    for n in needles:
        if _norm(n) in t:
            return _norm(n)
    return None


def _utilization(acct: Dict[str, Any]) -> Optional[float]:
    try:
        bal = float(acct.get("balance_owed"))
        lim = float(acct.get("credit_limit"))
        if lim and lim > 0:
            return bal / lim
    except Exception:
        pass
    return None


def _pick_t1(primary_hits: Dict[str, List[str]]) -> Optional[str]:
    for key in PRIORITY_T1:
        if key in primary_hits and primary_hits[key]:
            return key
    return None


def evaluate_account_problem(acct: Dict[str, Any]) -> Dict[str, Any]:
    reasons: List[str] = []
    supporting: Dict[str, Any] = {}
    primary_hits: Dict[str, List[str]] = {}
    repetition = 0

    scan_fields = {
        "account_status": acct.get("account_status"),
        "remarks": acct.get("creditor_remarks"),
        "description": acct.get("account_description"),
        "bureau_statuses": " ".join((acct.get("bureau_statuses") or {}).values()),
    }
    if ENABLE_TIER1_KEYWORDS:
        for label, tokens in TIER1_KEYWORDS.items():
            for field_name, raw in scan_fields.items():
                hit = _contains_any(raw, tokens)
                if hit:
                    reasons.append(f"{field_name}:{label}")
                    primary_hits.setdefault(label, []).append(field_name)
                    repetition += 1

    primary_issue = None
    tier = None
    if primary_hits:
        primary_issue = _pick_t1(primary_hits)
        tier = 1

    # Fallback detection for strong status keywords even when keyword lists are empty.
    if not primary_issue:
        for field_name, raw in scan_fields.items():
            text = _norm(raw)
            if re.search(r"charge[- ]?off", text):
                reasons.append(f"{field_name}:charge_off")
                primary_issue = "charge_off"
                tier = 1
                repetition += 1
                break
            if "collection" in text:
                reasons.append(f"{field_name}:collection")
                primary_issue = "collection"
                tier = 1
                repetition += 1
                break

    if not primary_issue:
        if ENABLE_TIER2_KEYWORDS:
            pstatus = _norm(acct.get("payment_status"))
            kw = _contains_any(pstatus, TIER2_KEYWORDS.get("serious_delinquency", []))
            if kw:
                reasons.append("payment_status:serious_delinquency")
                primary_issue = "serious_delinquency"
                tier = 2
                repetition += 1
        if not primary_issue and ENABLE_TIER2_NUMERIC:
            late = acct.get("late_payments") or {}
            max_bucket = 0
            for bureau, buckets in late.items():
                for k, v in (buckets or {}).items():
                    try:
                        days = int(k)
                        if days > max_bucket and v and int(v) > 0:
                            max_bucket = days
                    except Exception:
                        continue
            if max_bucket >= SERIOUS_DELINQUENCY_MIN_DPD:
                reasons.append(f"late_payments:{max_bucket}_dpd")
                primary_issue = "serious_delinquency"
                tier = 2
                repetition += 1

    util = _utilization(acct)
    if util is not None:
        supporting["utilization"] = round(util, 4)
        if util >= UTILIZATION_PROBLEM_THRESHOLD:
            reasons.append(f"utilization:>{int(UTILIZATION_PROBLEM_THRESHOLD*100)}%")

    if not primary_issue and ENABLE_TIER3_KEYWORDS:
        ar = _norm(acct.get("account_rating"))
        desc = _norm(acct.get("account_description"))
        t3_tokens = TIER3_KEYWORDS.get("potential_derogatory", [])
        t3_hit = _contains_any(ar, t3_tokens) or _contains_any(desc, t3_tokens)
        if t3_hit:
            reasons.append("account_rating:potential_derogatory")
            primary_issue = "potential_derogatory"
            tier = 3
            repetition += 1

    if acct.get("past_due_amount") is not None:
        supporting["past_due_amount"] = acct["past_due_amount"]

    conf = ConfidenceHint(
        tier=tier or 4,
        strongest_signal=primary_issue or "unknown",
        repetition_count=max(repetition, 0),
        latest_date_seen=None,
    )

    result = {
        "primary_issue": primary_issue or "unknown",
        "issue_types": [primary_issue] if primary_issue else [],
        "problem_reasons": reasons,
        "confidence": 1.0 if primary_issue else 0.0,
        "tier": tier or 0,
        "decision_source": "rules",
        "debug": {"supporting": supporting, "confidence_hint": asdict(conf)},
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
