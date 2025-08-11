"""Post-processing utilities for credit report analysis results."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Set
import re

from logic.utils.names_normalization import (
    normalize_creditor_name,
    normalize_bureau_name,
)


# ---------------------------------------------------------------------------
# Inquiry merging
# ---------------------------------------------------------------------------


def _merge_parser_inquiries(result: dict, parsed: List[dict]):
    """Merge parser-detected inquiries, preferring them over GPT output.

    Any inquiries present in ``parsed`` but missing from the AI result are
    injected with an ``advisor_comment`` note so downstream code can track the
    source.
    """
    cleaned: List[dict] = []
    seen = set()

    gpt_set = {
        (
            normalize_creditor_name(i.get("creditor_name")),
            i.get("date"),
            normalize_bureau_name(i.get("bureau")),
        )
        for i in result.get("inquiries", [])
    }

    for inq in parsed:
        key = (
            normalize_creditor_name(inq.get("creditor_name")),
            inq.get("date"),
            normalize_bureau_name(inq.get("bureau")),
        )
        if key in seen:
            continue
        entry = {
            "creditor_name": inq.get("creditor_name"),
            "date": inq.get("date"),
            "bureau": normalize_bureau_name(inq.get("bureau")),
        }
        if key not in gpt_set:
            entry["advisor_comment"] = "Detected by parser; missing from AI output"
        cleaned.append(entry)
        seen.add(key)

    for inq in result.get("inquiries", []):
        key = (
            normalize_creditor_name(inq.get("creditor_name")),
            inq.get("date"),
            normalize_bureau_name(inq.get("bureau")),
        )
        if key not in seen:
            cleaned.append(inq)
            seen.add(key)

    if cleaned:
        result["inquiries"] = cleaned
    elif "inquiries" in result:
        # Ensure field exists even if empty for downstream code
        result["inquiries"] = []


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
            "bureaus": list(bureaus.keys()),
            "status": "Unknown",
            "advisor_comment": "Detected by parser; missing from AI output",
            "late_payments": bureaus,
            "flags": ["Late Payments"],
        }
        result.setdefault("all_accounts", []).append(entry)
        print(f"[âš ï¸] Added missing account from parser: {entry['name']}")


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
        warnings.append("âš ï¸ No dispute/goodwill accounts found.")

    total_inquiries = analysis.get("summary_metrics", {}).get("total_inquiries")
    if isinstance(total_inquiries, list):
        if len(total_inquiries) > 50:
            warnings.append(
                "âš ï¸ Too many inquiries detected â€" may indicate parsing issue."
            )
    elif isinstance(total_inquiries, int):
        if total_inquiries > 50:
            warnings.append(
                "âš ï¸ Too many inquiries detected â€" may indicate parsing issue."
            )

    if not analysis.get("strategic_recommendations"):
        warnings.append("âš ï¸ No strategic recommendations provided.")

    for section in ["negative_accounts", "open_accounts_with_issues", "all_accounts"]:
        for account in analysis.get(section, []):
            comment = account.get("advisor_comment", "")
            if len(comment.split()) < 4:
                warnings.append(
                    f"âš ï¸ Advisor comment too short for account: {account.get('name')}"
                )

    if warnings:
        print("\n[!] ANALYSIS QA WARNINGS:")
        for warn in warnings:
            print(warn)

    return warnings
