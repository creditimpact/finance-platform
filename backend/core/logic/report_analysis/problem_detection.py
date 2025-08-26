import logging
from typing import Any, Dict, List, Mapping, Tuple

from backend.core.logic.utils.names_normalization import normalize_creditor_name

logger = logging.getLogger(__name__)

COLLECTOR_WHITELIST = {
    "palisades fu",
    "midland",
    "midland credit management",
    "portfolio recovery",
    "lvnv funding",
    "enhanced recovery",
    "cmre financial",
    "allied interstate",
    "portfolio rcvy",
    "asset acceptance",
    "transworld systems",
    "iqor",
    "ccs",
    "americollect",
    "cavalry",
    "firstsource",
    "national credit systems",
}

KEYWORDS = [
    "collection",
    "placed for collection",
    "charge off",
    "charged off",
    "derogatory",
    "repossession",
    "foreclosure",
    "collection agency",
    "collections",
    "bankruptcy",
    "judgment",
    "lien",
]


def _sum_late_payments(late_payments: Any) -> int:
    total = 0
    if isinstance(late_payments, Mapping):
        for bureau_vals in late_payments.values():
            if isinstance(bureau_vals, Mapping):
                for v in bureau_vals.values():
                    try:
                        total += int(v) or 0
                    except Exception:
                        continue
            else:
                try:
                    total += int(bureau_vals) or 0
                except Exception:
                    continue
    return total


def is_problematic(
    acc: Mapping[str, Any]
) -> Tuple[bool, List[str], str, Dict[str, Any]]:
    """Determine if an account is problematic based on simple evidence heuristics.

    Returns a tuple ``(is_problem, reasons, inferred_issue, context)``.
    ``context`` includes diagnostic info such as ``late_sum`` and ``keywords_hit``.
    """

    reasons: List[str] = []
    late_sum = _sum_late_payments(acc.get("late_payments"))
    if late_sum > 0:
        reasons.append(f"late_payment_sum:{late_sum}")

    text_slots = " ".join(
        str(acc.get(field, "") or "")
        for field in ["remarks", "status", "bureau_statuses", "account_status"]
    ).lower()
    keywords_hit = [kw for kw in KEYWORDS if kw in text_slots]
    if keywords_hit:
        reasons.extend([f"keyword:{kw}" for kw in keywords_hit])

    norm_name = acc.get("normalized_name") or normalize_creditor_name(
        acc.get("name", "")
    )
    is_collector = norm_name in COLLECTOR_WHITELIST
    if is_collector:
        reasons.append("name_matches_collector")

    inferred_issue = "unknown"
    if is_collector or any(
        kw
        in {
            "collection",
            "placed for collection",
            "charge off",
            "charged off",
            "collection agency",
            "collections",
        }
        for kw in keywords_hit
    ):
        inferred_issue = "collection"
    elif late_sum > 0:
        inferred_issue = "late_payment"

    context = {
        "late_sum": late_sum,
        "keywords_hit": keywords_hit,
        "is_collector": is_collector,
    }
    return bool(reasons), reasons, inferred_issue, context
