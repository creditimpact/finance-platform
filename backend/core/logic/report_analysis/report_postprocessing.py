"""Post-processing utilities for credit report analysis results."""

from __future__ import annotations

import re
from hashlib import sha1
from typing import Any, Dict, List, Mapping, Set
from uuid import uuid4

from backend.core.logic.utils.names_normalization import (
    normalize_bureau_name,
    normalize_creditor_name,
)
from backend.core.logic.utils.norm import normalize_heading
from rapidfuzz import fuzz

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


def _fuzzy_match(name: str, choices: Set[str]) -> str | None:
    """Return best fuzzy match for *name* within *choices* when >= 0.8."""

    best_score = 0.0
    best: str | None = None
    for cand in choices:
        score = fuzz.WRatio(name, cand) / 100.0
        if score > best_score:
            best_score = score
            best = cand
    if best_score >= 0.8:
        return best
    return None


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

    def _clean_number(value: str) -> str | None:
        """Normalize ``value`` if it contains digits; otherwise return ``None``."""

        if not re.search(r"\d", value or ""):
            return None
        return re.sub(r"[\s-]", "", value)

    # Sanitize any pre-existing account number fields on the root object
    for field in ("account_number_raw", "account_number", "account_number_masked"):
        val = acc.get(field)
        if isinstance(val, str):
            cleaned = _clean_number(val)
            if cleaned:
                if field == "account_number":
                    cleaned = re.sub(r"\D", "", cleaned)
                acc[field] = cleaned
            else:
                acc.pop(field, None)

    # Sanitize bureau level account numbers
    for info in acc.get("bureaus", []) or []:
        if not isinstance(info, dict):
            continue
        for field in ("account_number_raw", "account_number", "account_number_masked"):
            val = info.get(field)
            if not isinstance(val, str):
                continue
            cleaned = _clean_number(val)
            if cleaned:
                if field == "account_number":
                    cleaned = re.sub(r"\D", "", cleaned)
                info[field] = cleaned
            else:
                info.pop(field, None)

    # Derive a last4 account number from any available account number field
    acct_num = (
        acc.get("account_number")
        or acc.get("account_number_masked")
        or acc.get("account_number_raw")
    )
    if not acct_num:
        for info in acc.get("bureaus", []) or []:
            if not isinstance(info, dict):
                continue
            acct_num = (
                info.get("account_number")
                or info.get("account_number_masked")
                or info.get("account_number_raw")
            )
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

    # Derive a stable fingerprint when no account number is available
    if "account_number_last4" not in acc:
        late = acc.get("late_payments") or {}
        seed = (
            f"{acc['normalized_name']}|{acc.get('date_opened')}|"
            f"{','.join(sorted(late.keys()))}"
        )
        if not acc.get("date_opened") and not late:
            seed += f"|{acc.get('original_creditor')}|{acc.get('balance')}"
        acc["account_fingerprint"] = sha1(seed.encode()).hexdigest()[:8]

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
            normalize_heading(i.get("creditor_name")),
            i.get("date"),
            normalize_bureau_name(i.get("bureau")),
        )
        for i in result.get("inquiries", [])
    }
    gpt_by_db: Dict[tuple[str | None, str], Set[str]] = {}
    for norm_name, date, bureau in gpt_set:
        gpt_by_db.setdefault((date, bureau), set()).add(norm_name)

    raw_keys: Set[str] = set(raw_map.keys())

    def _lookup_raw(name: str) -> str | None:
        if name in raw_map:
            return raw_map[name]
        match = _fuzzy_match(name, raw_keys)
        if match:
            return raw_map.get(match)
        return None

    for inq in parsed:
        key_name = normalize_heading(inq.get("creditor_name"))
        date = inq.get("date")
        bureau = normalize_bureau_name(inq.get("bureau"))
        names = gpt_by_db.get((date, bureau), set())
        match = key_name if key_name in names else _fuzzy_match(key_name, names)
        key = (match or key_name, date, bureau)
        if key in seen:
            continue
        creditor_name = _lookup_raw(key_name) or inq.get("creditor_name") or str(uuid4())
        entry = {
            "creditor_name": creditor_name,
            "date": date,
            "bureau": bureau,
        }
        if not match:
            entry["advisor_comment"] = "Detected by parser; missing from AI output"
        cleaned.append(entry)
        seen.add(key)

    for inq in result.get("inquiries", []):
        key_name = normalize_heading(inq.get("creditor_name"))
        key = (
            key_name,
            inq.get("date"),
            normalize_bureau_name(inq.get("bureau")),
        )
        if key not in seen:
            creditor_name = _lookup_raw(key_name) or inq.get("creditor_name") or str(uuid4())
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
    heading_keys: Set[str] = set(headings.keys())
    for sec in sections:
        for acc in result.get(sec, []):
            raw = acc.get("name", "")
            norm = normalize_heading(raw)
            match = norm if norm in headings else _fuzzy_match(norm, heading_keys)
            if match:
                if headings[match] != raw:
                    acc["name"] = headings[match]
                seen.add(match)

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

    co_bureaus: Set[str] = set(acc.get("co_bureaus", []) or [])

    # Inspect explicit late payment counts from the parser or AI output and
    # detect charge-off markers present in the late grids.
    late_map = acc.get("late_payments") or {}
    has_co_marker = acc.get("has_co_marker", False)
    if isinstance(late_map, dict):
        # ``late_payments`` may be either a mapping of bureau -> counts or a
        # direct mapping of day buckets -> counts.  Any positive count should
        # trigger a ``late_payment`` issue type.  ``CO`` tokens indicate a
        # charge-off marker that should be preserved for downstream logic.
        for bureau, bureau_vals in late_map.items():
            if isinstance(bureau_vals, dict):
                for bucket, count in bureau_vals.items():
                    try:
                        if int(count) > 0:
                            issue_types.add("late_payment")
                            if bucket.upper() == "CO":
                                has_co_marker = True
                                co_bureaus.add(bureau)
                    except (TypeError, ValueError):
                        continue
            else:
                try:
                    if int(bureau_vals) > 0:
                        issue_types.add("late_payment")
                except (TypeError, ValueError):
                    continue

    for bureau, hist in (acc.get("grid_history_raw") or {}).items():
        if isinstance(hist, str) and re.search(r"\bCO\b", hist, re.I):
            has_co_marker = True
            co_bureaus.add(bureau)

    if has_co_marker or co_bureaus:
        acc["has_co_marker"] = True
        if co_bureaus:
            acc["co_bureaus"] = sorted(co_bureaus)
        flags_list = acc.setdefault("flags", [])
        if "Charge-Off" not in flags_list:
            flags_list.append("Charge-Off")

    # Bureau detail lines may indicate charge-off or collection statuses even
    # when no explicit late history is available. Inspect each bureau's
    # ``account_status``, ``account_description`` and ``payment_status`` fields
    # for key phrases. If found, elevate the issue type and treat it as evidence
    # of a charge-off/collection. Past-due amounts without such markers should
    # still yield a ``late_payment`` issue.
    details_map = acc.get("bureau_details") or {}
    detail_co = False
    detail_past_due = False
    for bureau, fields in details_map.items():
        if not isinstance(fields, dict):
            continue
        text = " ".join(
            str(fields.get(k) or "")
            for k in ("account_status", "account_description", "payment_status")
        ).lower()
        if re.search(r"collection", text):
            issue_types.add("collection")
            co_bureaus.add(bureau)
            detail_co = True
        if re.search(r"charge[ -]?off|bad debt", text):
            issue_types.add("charge_off")
            co_bureaus.add(bureau)
            detail_co = True
        if re.search(r"repossession", text):
            issue_types.add("repossession")
        past_due_val = fields.get("past_due_amount")
        try:
            if float(str(past_due_val).replace(",", "")) > 0:
                detail_past_due = True
        except (TypeError, ValueError):
            pass
    if detail_co:
        acc["has_co_marker"] = True
        acc["co_bureaus"] = sorted(co_bureaus)
    elif detail_past_due:
        issue_types.add("late_payment")

    remarks = acc.get("remarks")
    if isinstance(remarks, str):
        rl = remarks.lower()
        acc["remarks_contains_co"] = (
            "charge" in rl and "off" in rl
        ) or "collection" in rl

    # Aggregate any status-like text from the account and its bureau entries
    status_parts = [
        str(acc.get("status") or ""),
        str(acc.get("account_status") or ""),
        str(acc.get("payment_status") or ""),
        str(acc.get("remarks") or ""),
    ]
    for val in (acc.get("payment_statuses") or {}).values():
        status_parts.append(str(val or ""))
    status_text_hits: list[str] = []
    for bureau, txt in (acc.get("status_texts") or {}).items():
        text = str(txt or "")
        status_parts.append(text)
        low = text.lower()
        matches: list[str] = []
        if re.search(r"\bcharge[-\s]?off\b|\bchargeoff\b", low):
            matches.append("Chargeoff")
        if re.search(r"\bcollection(s)?\b", low):
            matches.append("Collection")
        if matches:
            bureau_name = normalize_bureau_name(bureau)
            status_text_hits.append(f"{bureau_name}: {'/'.join(matches)}")
    for key, val in acc.items():
        if "history" in key:
            status_parts.append(str(val or ""))
    for info in acc.get("bureaus", []) or []:
        if isinstance(info, dict):
            status_parts.append(str(info.get("status") or ""))
            status_parts.append(str(info.get("account_status") or ""))
            status_parts.append(str(info.get("payment_status") or ""))
            status_parts.append(str(info.get("remarks") or ""))
            for key, val in info.items():
                if "history" in key:
                    status_parts.append(str(val or ""))
    status_text = " ".join(status_parts).lower()
    status_clean = status_text.replace("-", " ").replace("_", " ")

    flags = [f.lower().replace("-", " ") for f in acc.get("flags", [])]

    if "bankrupt" in status_clean or any("bankrupt" in f for f in flags):
        issue_types.add("bankruptcy")

    # Look for charge-off and collection keywords in status text and flags
    co_grid = bool(re.search(r"\bco\b", status_clean))
    has_charge_off = bool(
        has_co_marker
        or re.search(r"charge\s*off|charged\s*off|chargeoff", status_clean)
        or any("charge off" in f for f in flags)
        or co_grid
    )
    has_collection = bool(
        re.search(r"collection", status_clean) or any("collection" in f for f in flags)
    )
    if has_charge_off or has_collection:
        acc["has_co_marker"] = True
    if has_charge_off:
        issue_types.add("charge_off")
    if has_collection:
        issue_types.add("collection")

    if (
        "repossession" in status_clean
        or "repossess" in status_clean
        or any("repossession" in f or "repossess" in f for f in flags)
    ):
        issue_types.add("repossession")

    if "foreclosure" in status_clean or any("foreclosure" in f for f in flags):
        issue_types.add("foreclosure")

    if "collection" in issue_types and "charge_off" in issue_types:
        primary = "collection"
    else:
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

    if status_text_hits:
        evidence = acc.setdefault("evidence", {})
        existing_hits = evidence.setdefault("status_text_hits", [])
        for hit in status_text_hits:
            if hit not in existing_hits:
                existing_hits.append(hit)


def _inject_missing_late_accounts(
    result: dict, history: dict, raw_map: dict, grid_map: dict | None = None
) -> None:
    """Add accounts detected by the parser but missing from the AI output."""
    existing = {
        normalize_creditor_name(acc.get("name", ""))
        for acc in result.get("all_accounts", [])
    }

    grid_map = grid_map or {}

    for norm_name, bureaus in history.items():
        if norm_name in existing:
            continue

        flags: List[str] = ["Late Payments"]
        histories = grid_map.get(norm_name, {})
        co_bureaus = [
            b
            for b, txt in histories.items()
            if isinstance(txt, str) and re.search(r"\bCO\b", txt, re.I)
        ]
        if co_bureaus:
            flags.append("Charge-Off")

        entry = {
            "name": raw_map.get(norm_name, norm_name),
            "late_payments": bureaus,
            "status": "Delinquent",
            "advisor_comment": "Late payments detected by parser; AI unavailable",
            "flags": flags,
            "source_stage": "parser_aggregated",
        }
        if histories:
            entry["grid_history_raw"] = histories
        if co_bureaus:
            entry["co_bureaus"] = sorted(co_bureaus)
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
