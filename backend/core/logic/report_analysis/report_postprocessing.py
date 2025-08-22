"""Post-processing utilities for credit report analysis results."""

from __future__ import annotations

import re
from uuid import uuid4
from typing import Any, Dict, List, Mapping, Set

from backend.core.logic.utils.names_normalization import (
    normalize_bureau_name,
    normalize_creditor_name,
)

# Ordered by descending severity
ISSUE_SEVERITY = [
    "bankruptcy",
    "charge_off",
    "collection",
    "repossession",
    "foreclosure",
    "late_payment",
]

ISSUE_TEXT: Mapping[str, tuple[str, str]] = {
    "bankruptcy": ("Bankruptcy", "Bankruptcy reported"),
    "charge_off": ("Charge Off", "Account charged off"),
    "collection": ("Collection", "Account in collection"),
    "repossession": ("Repossession", "Account repossessed"),
    "foreclosure": ("Foreclosure", "Account in foreclosure"),
    "late_payment": ("Delinquent", "Late payments detected"),
}


def pick_primary_issue(issue_set: set[str]) -> str:
    """Select the most severe issue present in ``issue_set``.

    Severity is determined by the ordering in ``ISSUE_SEVERITY``. If no
    recognized issue is found, ``"unknown"`` is returned.
    """

    for tag in ISSUE_SEVERITY:
        if tag in issue_set:
            return tag
    return "unknown"


def enrich_account_metadata(acc: dict[str, Any]) -> dict[str, Any]:
    """Populate standardized metadata for a problematic account.

    The enrichment ensures downstream components such as ``BureauPayload``
    receive a consistent set of fields regardless of whether the account
    originated from the AI analysis or was synthesized from parser signals.
    The function mutates ``acc`` in place and also returns it for convenience.
    """

    # Normalized creditor name for reliable matching
    name = acc.get("name", "")
    acc["normalized_name"] = normalize_creditor_name(name)

    # Derive a last4 account number from any available account number field
    acct_num = acc.get("account_number") or acc.get("account_number_masked")
    if not acct_num:
        for info in acc.get("bureaus", []) or []:
            if not isinstance(info, dict):
                continue
            acct_num = info.get("account_number") or info.get("account_number_masked")
            if acct_num:
                break
    if isinstance(acct_num, str):
        digits = re.sub(r"\D", "", acct_num)
        if digits:
            acc["account_number_last4"] = digits[-4:]

    # Pull common metadata from bureau entries if missing on the root object
    meta_fields = [
        "original_creditor",
        "account_type",
        "balance",
        "past_due",
        "date_opened",
        "date_closed",
        "last_activity",
    ]
    for field in meta_fields:
        if acc.get(field) not in (None, ""):
            continue
        for info in acc.get("bureaus", []) or []:
            if isinstance(info, dict) and info.get(field) not in (None, ""):
                acc[field] = info[field]
                break

    # Build a distilled status per bureau when bureau level info is available
    statuses: dict[str, str] = {}
    for info in acc.get("bureaus", []) or []:
        if not isinstance(info, dict):
            continue
        bureau = info.get("bureau") or info.get("name")
        if not bureau:
            continue
        status_text = str(
            info.get("status") or info.get("account_status") or ""
        ).lower()
        short = ""
        if "charge off" in status_text or "collection" in status_text:
            short = "Collection/Chargeoff"
        elif "120" in status_text:
            short = "120d late"
        elif "90" in status_text:
            short = "90d late"
        elif "60" in status_text:
            short = "60d late"
        elif "30" in status_text:
            short = "30d late"
        elif "open" in status_text or "current" in status_text:
            short = "Open/Current"
        else:
            late_map = info.get("late_payments") or {}
            for days in ["120", "90", "60", "30"]:
                if int(late_map.get(days, 0)) > 0:
                    short = f"{days}d late"
                    break
            if not short:
                short = status_text.title() if status_text else ""
        statuses[bureau] = short
    if statuses:
        acc["bureau_statuses"] = statuses

    # Ensure a source stage marker exists
    acc.setdefault("source_stage", "ai_final")

    # Append any evidence flags (e.g., tri-merge mismatches)
    tri_info = acc.get("tri_merge") or {}
    evidence_flags = list(tri_info.get("mismatch_types", []))
    evidence = tri_info.get("evidence", {})
    evidence_flags.extend(
        evidence.get("flags", []) if isinstance(evidence, dict) else []
    )
    if evidence_flags:
        existing = acc.setdefault("flags", [])
        for flag in evidence_flags:
            if flag not in existing:
                existing.append(flag)

    return acc


# ---------------------------------------------------------------------------
# Inquiry merging
# ---------------------------------------------------------------------------


def _merge_parser_inquiries(
    result: dict, parsed: List[dict], raw_map: Mapping[str, str] | None = None
):
    """Merge parser-detected inquiries, preferring them over GPT output.

    Any inquiries present in ``parsed`` but missing from the AI result are
    injected with an ``advisor_comment`` note so downstream code can track the
    source. ``raw_map`` allows restoration of the human-readable creditor label
    when available.
    """
    cleaned: List[dict] = []
    seen = set()
    raw_map = raw_map or {}

    gpt_set = {
        (
            normalize_creditor_name(i.get("creditor_name")),
            i.get("date"),
            normalize_bureau_name(i.get("bureau")),
        )
        for i in result.get("inquiries", [])
    }

    for inq in parsed:
        key_name = normalize_creditor_name(inq.get("creditor_name"))
        key = (
            key_name,
            inq.get("date"),
            normalize_bureau_name(inq.get("bureau")),
        )
        if key in seen:
            continue
        creditor_name = raw_map.get(key_name) or inq.get("creditor_name") or str(uuid4())
        entry = {
            "creditor_name": creditor_name,
            "date": inq.get("date"),
            "bureau": normalize_bureau_name(inq.get("bureau")),
        }
        if key not in gpt_set:
            entry["advisor_comment"] = "Detected by parser; missing from AI output"
        cleaned.append(entry)
        seen.add(key)

    for inq in result.get("inquiries", []):
        key_name = normalize_creditor_name(inq.get("creditor_name"))
        key = (
            key_name,
            inq.get("date"),
            normalize_bureau_name(inq.get("bureau")),
        )
        if key not in seen:
            creditor_name = raw_map.get(key_name) or inq.get("creditor_name") or str(uuid4())
            inq["creditor_name"] = creditor_name
            cleaned.append(inq)
            seen.add(key)

    if cleaned:
        result["inquiries"] = cleaned
    elif "inquiries" in result:
        # Ensure field exists even if empty for downstream code
        result["inquiries"] = []


# ---------------------------------------------------------------------------
# Account heading reconciliation
# ---------------------------------------------------------------------------


def _reconcile_account_headings(result: dict, headings: Mapping[str, str]) -> None:
    """Align AI account names with parser-detected headings."""

    if not headings:
        return

    seen = set()
    sections = [
        "all_accounts",
        "negative_accounts",
        "open_accounts_with_issues",
        "positive_accounts",
        "high_utilization_accounts",
    ]
    for sec in sections:
        for acc in result.get(sec, []):
            raw = acc.get("name", "")
            norm = normalize_creditor_name(raw)
            if norm in headings:
                if headings[norm] != raw:
                    acc["name"] = headings[norm]
                seen.add(norm)

    for norm, raw in headings.items():
        if norm not in seen:
            print(
                f"[WARN] Parser detected account heading '{raw}' missing from AI output"
            )


# ---------------------------------------------------------------------------
# Late payment utilities
# ---------------------------------------------------------------------------


def _sanitize_late_counts(history: Dict[str, Dict[str, Dict[str, int]]]) -> None:
    """Remove unrealistic late payment numbers from parsed history."""
    for acc, bureaus in list(history.items()):
        for bureau, counts in list(bureaus.items()):
            for key, val in list(counts.items()):
                if val > 12:
                    print(
                        f"[~] Dropping unrealistic count {val}x{key} for {acc} ({bureau})"
                    )
                    counts.pop(key)
            if not counts:
                bureaus.pop(bureau)
        if not bureaus:
            history.pop(acc)


def _cleanup_unverified_late_text(result: dict, verified: Set[str]):
    """Remove GPT late references for accounts without verified history."""

    def clean(acc: dict):
        norm = normalize_creditor_name(acc.get("name", ""))
        if norm in verified:
            return
        if "flags" in acc:
            acc["flags"] = [f for f in acc["flags"] if "late" not in f.lower()]
            if not acc["flags"]:
                acc.pop("flags")
        comment = acc.get("advisor_comment")
        if comment and re.search(r"late|delinqu", comment, re.I):
            acc.pop("advisor_comment", None)

    for sec in [
        "all_accounts",
        "negative_accounts",
        "open_accounts_with_issues",
        "positive_accounts",
        "high_utilization_accounts",
    ]:
        for a in result.get(sec, []):
            clean(a)


def _assign_issue_types(acc: dict) -> None:
    """Derive ``issue_types`` and fallback metadata for an account.

    Inspects ``late_payments``, ``status`` and ``flags`` to infer issue
    categories.  Populates ``acc['issue_types']`` and provides default
    ``status`` and ``advisor_comment`` values when missing.
    """

    issue_types: Set[str] = set(acc.get("issue_types", []))

    # Aggregate any status-like text from the account and its bureau entries
    status_parts = [
        str(acc.get("status") or ""),
        str(acc.get("account_status") or ""),
    ]
    for info in acc.get("bureaus", []) or []:
        if isinstance(info, dict):
            status_parts.append(str(info.get("status") or ""))
            status_parts.append(str(info.get("account_status") or ""))
    status_text = " ".join(status_parts).lower()
    status_clean = status_text.replace("-", " ")

    flags = [f.lower().replace("-", " ") for f in acc.get("flags", [])]

    # Inspect explicit late payment counts from the parser or AI output
    late_map = acc.get("late_payments") or {}
    if isinstance(late_map, dict):
        # ``late_payments`` may be either a mapping of bureau -> counts
        # or a direct mapping of day buckets -> counts.  Any positive count
        # should trigger a ``late_payment`` issue type.
        for bureau_vals in late_map.values():
            # Handle direct maps of day -> count
            if not isinstance(bureau_vals, dict):
                try:
                    if int(bureau_vals) > 0:
                        issue_types.add("late_payment")
                        break
                except (TypeError, ValueError):
                    continue
                continue
            for count in bureau_vals.values():
                try:
                    if int(count) > 0:
                        issue_types.add("late_payment")
                        break
                except (TypeError, ValueError):
                    continue
            if "late_payment" in issue_types:
                break

    if "bankrupt" in status_clean or any("bankrupt" in f for f in flags):
        issue_types.add("bankruptcy")

    # Look for charge-off and collection keywords in status text and flags
    if (
        re.search(r"charge\s*off|charged\s*off|chargeoff", status_clean)
        or any("charge off" in f for f in flags)
    ):
        issue_types.add("charge_off")

    if (
        re.search(r"collection", status_clean)
        or any("collection" in f for f in flags)
    ):
        issue_types.add("collection")

    if (
        "repossession" in status_clean
        or "repossess" in status_clean
        or any("repossession" in f or "repossess" in f for f in flags)
    ):
        issue_types.add("repossession")

    if "foreclosure" in status_clean or any("foreclosure" in f for f in flags):
        issue_types.add("foreclosure")

    primary = pick_primary_issue(issue_types)
    acc["primary_issue"] = primary

    severity_index = {t: i for i, t in enumerate(ISSUE_SEVERITY)}
    sorted_all = sorted(
        issue_types, key=lambda t: severity_index.get(t, len(ISSUE_SEVERITY))
    )
    if primary != "unknown" and primary in issue_types:
        sorted_types = [primary] + [t for t in sorted_all if t != primary]
    else:
        sorted_types = sorted_all
    acc["issue_types"] = sorted_types

    status, comment = ISSUE_TEXT.get(primary, (None, None))
    if status:
        acc["status"] = status
    if comment:
        acc["advisor_comment"] = comment


def _inject_missing_late_accounts(result: dict, history: dict, raw_map: dict) -> None:
    """Add accounts detected by the parser but missing from the AI output."""
    existing = {
        normalize_creditor_name(acc.get("name", ""))
        for acc in result.get("all_accounts", [])
    }

    for norm_name, bureaus in history.items():
        if norm_name in existing:
            continue

        entry = {
            "name": raw_map.get(norm_name, norm_name),
            "late_payments": bureaus,
            "status": "Delinquent",
            "advisor_comment": "Late payments detected by parser; AI unavailable",
            "flags": ["Late Payments"],
            "source_stage": "parser_aggregated",
        }
        _assign_issue_types(entry)
        enriched = enrich_account_metadata(entry)
        result.setdefault("all_accounts", []).append(enriched)
        if enriched.get("issue_types"):
            result.setdefault("negative_accounts", []).append(enriched.copy())
        print(
            f"[WARN] Aggregated missing account from parser: {entry['name']} "
            f"bureaus={list(bureaus.keys())}"
        )


# ---------------------------------------------------------------------------
# Analysis sanity checks
# ---------------------------------------------------------------------------


def validate_analysis_sanity(analysis: Mapping[str, Any]) -> List[str]:
    """Run lightweight sanity checks on the final analysis structure.

    Returns a list of warning messages.  The function prints them as a side
    effect to assist with manual debugging.
    """
    warnings: List[str] = []

    if not analysis.get("negative_accounts") and not analysis.get(
        "open_accounts_with_issues"
    ):
        warnings.append("WARN No dispute/goodwill accounts found.")

    total_inquiries = analysis.get("summary_metrics", {}).get("total_inquiries")
    if isinstance(total_inquiries, list):
        if len(total_inquiries) > 50:
            warnings.append(
                "WARN Too many inquiries detected - may indicate parsing issue."
            )
    elif isinstance(total_inquiries, int):
        if total_inquiries > 50:
            warnings.append(
                "WARN Too many inquiries detected - may indicate parsing issue."
            )

    if not analysis.get("strategic_recommendations"):
        warnings.append("WARN No strategic recommendations provided.")

    for section in ["negative_accounts", "open_accounts_with_issues", "all_accounts"]:
        for account in analysis.get(section, []):
            comment = account.get("advisor_comment", "")
            if len(comment.split()) < 4:
                warnings.append(
                    f"WARN Advisor comment too short for account: {account.get('name')}"
                )

    if warnings:
        print("\n[!] ANALYSIS QA WARNINGS:")
        for warn in warnings:
            print(warn)

    return warnings
