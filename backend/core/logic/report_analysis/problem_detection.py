from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional
import re

from backend.config import (
    UTILIZATION_PROBLEM_THRESHOLD,
    SERIOUS_DELINQUENCY_MIN_DPD,
    TIER1_KEYWORDS,
    TIER2_KEYWORDS,
    TIER3_KEYWORDS,
)

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

    if not primary_issue:
        pstatus = _norm(acct.get("payment_status"))
        kw = _contains_any(pstatus, TIER2_KEYWORDS["serious_delinquency"])
        if kw:
            reasons.append("payment_status:serious_delinquency")
            primary_issue = "serious_delinquency"
            tier = 2
            repetition += 1
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
        if not primary_issue and max_bucket >= SERIOUS_DELINQUENCY_MIN_DPD:
            reasons.append(f"late_payments:{max_bucket}_dpd")
            primary_issue = "serious_delinquency"
            tier = 2
            repetition += 1

    util = _utilization(acct)
    if util is not None:
        supporting["utilization"] = round(util, 4)
        if util >= UTILIZATION_PROBLEM_THRESHOLD:
            reasons.append(f"utilization:>{int(UTILIZATION_PROBLEM_THRESHOLD*100)}%")

    if not primary_issue:
        ar = _norm(acct.get("account_rating"))
        desc = _norm(acct.get("account_description"))
        t3_hit = _contains_any(ar, TIER3_KEYWORDS["potential_derogatory"]) or _contains_any(
            desc, TIER3_KEYWORDS["potential_derogatory"]
        )
        if t3_hit:
            reasons.append("account_rating:potential_derogatory")
            primary_issue = "potential_derogatory"
            tier = 3
            repetition += 1

    if acct.get("past_due_amount") is not None:
        supporting["past_due_amount"] = acct["past_due_amount"]

    is_problem = primary_issue in {
        "collection",
        "charge_off",
        "bankruptcy",
        "foreclosure",
        "judgment",
        "tax_lien",
        "serious_delinquency",
        "potential_derogatory",
    }

    conf = ConfidenceHint(
        tier=tier or 4,
        strongest_signal=primary_issue or "unknown",
        repetition_count=max(repetition, 0),
        latest_date_seen=None,
    )

    return {
        "is_problem": bool(is_problem and tier in {1, 2, 3}),
        "primary_issue": primary_issue or "unknown",
        "problem_reasons": reasons,
        "confidence_hint": asdict(conf),
        "supporting": supporting,
        "unknown_fields": [],
    }
