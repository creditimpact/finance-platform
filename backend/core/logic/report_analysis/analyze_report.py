"""High-level orchestration for analyzing credit reports.

This module wires together parsing utilities, prompt generation/AI calls,
and a suite of post-processing helpers. Historically all of this logic lived
in a single file which made the responsibilities difficult to test and
reason about. The functionality has been split into dedicated modules:

- :mod:`backend.core.logic.report_analysis.report_parsing`
- :mod:`backend.core.logic.report_analysis.report_prompting`
- :mod:`backend.core.logic.report_analysis.report_postprocessing`
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from backend.core.logic.utils.inquiries import extract_inquiries
from backend.core.logic.utils.names_normalization import normalize_creditor_name
from backend.core.logic.utils.text_parsing import (
    enforce_collection_status,
    extract_account_headings,
    extract_late_history_blocks,
)
from backend.core.services.ai_client import AIClient, get_ai_client

from .report_parsing import extract_text_from_pdf
from .report_postprocessing import (
    _cleanup_unverified_late_text,
    _inject_missing_late_accounts,
    _merge_parser_inquiries,
    _sanitize_late_counts,
    _reconcile_account_headings,
    validate_analysis_sanity,
)
from .report_prompting import (
    ANALYSIS_PROMPT_VERSION,
    ANALYSIS_SCHEMA_VERSION,
    call_ai_analysis,
)

# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def analyze_credit_report(
    pdf_path,
    output_json_path,
    client_info,
    ai_client: AIClient | None = None,
    run_ai: bool = True,
    *,
    request_id: str,
):
    """Analyze ``pdf_path`` and write structured analysis to ``output_json_path``."""
    ai_client = ai_client or get_ai_client()
    text = extract_text_from_pdf(pdf_path)
    if not text.strip():
        raise ValueError("[ERROR] No text extracted from PDF")

    headings = extract_account_headings(text)
    heading_map = {norm: raw for norm, raw in headings}

    def detected_late_phrases(txt: str) -> bool:
        return bool(re.search(r"late|past due", txt, re.I))

    raw_goal = client_info.get("goal", "").strip().lower()
    if raw_goal in ["", "not specified", "improve credit", "repair credit"]:
        strategic_context = (
            "Improve credit score significantly within the next 3-6 months using strategies such as authorized users, "
            "credit building tools, and removal of negative items."
        )
    else:
        strategic_context = client_info.get("goal", "Not specified")

    is_identity_theft = client_info.get("is_identity_theft", False)
    doc_fingerprint = hashlib.sha256(
        f"{text}|{ANALYSIS_PROMPT_VERSION}|{ANALYSIS_SCHEMA_VERSION}".encode(
            "utf-8"
        )
    ).hexdigest()
    if run_ai:
        result = call_ai_analysis(
            text,
            is_identity_theft,
            Path(output_json_path),
            ai_client=ai_client,
            strategic_context=strategic_context,
            request_id=request_id,
            doc_fingerprint=doc_fingerprint,
        )
    else:
        result = {
            "negative_accounts": [],
            "open_accounts_with_issues": [],
            "positive_accounts": [],
            "high_utilization_accounts": [],
            "all_accounts": [],
            "inquiries": [],
        }

    result["prompt_version"] = ANALYSIS_PROMPT_VERSION
    result["schema_version"] = ANALYSIS_SCHEMA_VERSION

    _reconcile_account_headings(result, heading_map)

    parsed_inquiries = extract_inquiries(text)
    if parsed_inquiries:
        print(f"[INFO] Parser found {len(parsed_inquiries)} inquiries in text.")
    else:
        print("[WARN] Parser did not find any inquiries in the report text.")

    if run_ai and result.get("inquiries"):
        print(f"[INFO] GPT found {len(result['inquiries'])} inquiries:")
        for inq in result["inquiries"]:
            print(f"  * {inq['creditor_name']} - {inq['date']} ({inq['bureau']})")
    elif run_ai:
        print("[WARN] No inquiries found in GPT result.")
    else:
        result["inquiries"] = parsed_inquiries

    try:
        account_names = {acc.get("name", "") for acc in result.get("all_accounts", [])}
        history_all, raw_map = extract_late_history_blocks(text, return_raw_map=True)
        _sanitize_late_counts(history_all)
        history = extract_late_history_blocks(text, account_names)
        _sanitize_late_counts(history)

        if history:
            print(f"[INFO] Found {len(history)} late payment block(s):")
            for creditor, bureaus in history.items():
                print(
                    f"[INFO] Detected late payments for: '{creditor.title()}' -> {bureaus}"
                )
        else:
            print("[ERROR] No late payment history blocks detected.")

        existing_norms = set()
        for acc in result.get("all_accounts", []):
            raw_name = acc.get("name", "")
            norm = normalize_creditor_name(raw_name)
            if raw_name and norm != raw_name.lower().strip():
                print(f"[~] Normalized account heading '{raw_name}' -> '{norm}'")
            existing_norms.add(norm)
            if norm in history:
                acc["late_payments"] = history[norm]
                if any(
                    v >= 1 for vals in history[norm].values() for v in vals.values()
                ):
                    acc.setdefault("flags", []).append("Late Payments")
                    status_text = (
                        str(acc.get("status") or acc.get("account_status") or "")
                        .strip()
                        .lower()
                    )
                    if status_text == "closed":
                        acc["goodwill_on_closed"] = True

        for section in [
            "negative_accounts",
            "open_accounts_with_issues",
            "positive_accounts",
            "high_utilization_accounts",
        ]:
            for acc in result.get(section, []):
                raw_name = acc.get("name", "")
                norm = normalize_creditor_name(raw_name)
                if raw_name and norm != raw_name.lower().strip():
                    print(f"[~] Normalized account heading '{raw_name}' -> '{norm}'")
                if norm in history:
                    acc["late_payments"] = history[norm]
                    if any(
                        v >= 1 for vals in history[norm].values() for v in vals.values()
                    ):
                        acc.setdefault("flags", []).append("Late Payments")
                        if section not in [
                            "negative_accounts",
                            "open_accounts_with_issues",
                        ]:
                            acc["goodwill_candidate"] = True
                        status_text = (
                            str(acc.get("status") or acc.get("account_status") or "")
                            .strip()
                            .lower()
                        )
                        if status_text == "closed":
                            acc["goodwill_on_closed"] = True

        for raw_norm, bureaus in history_all.items():
            linked = raw_norm in history
            if linked:
                print(
                    f"[INFO] Linked late payment block '{raw_map.get(raw_norm, raw_norm)}' to account '{raw_norm.title()}'"
                )
            else:
                snippet = raw_map.get(raw_norm, raw_norm)
                print(f"[WARN] Unlinked late-payment block detected near: '{snippet}'")

        # Remove any late_payment fields that were not verified by parser
        verified_names = set(history.keys())

        def strip_unverified(acc_list):
            for acc in acc_list:
                norm = normalize_creditor_name(acc.get("name", ""))
                if "late_payments" in acc and norm not in verified_names:
                    acc.pop("late_payments", None)

        for sec in [
            "all_accounts",
            "negative_accounts",
            "open_accounts_with_issues",
            "positive_accounts",
            "high_utilization_accounts",
        ]:
            strip_unverified(result.get(sec, []))

        _cleanup_unverified_late_text(result, verified_names)

        _inject_missing_late_accounts(result, history_all, raw_map)

        _merge_parser_inquiries(result, parsed_inquiries)

        for section in [
            "all_accounts",
            "negative_accounts",
            "open_accounts_with_issues",
            "positive_accounts",
            "high_utilization_accounts",
        ]:
            for acc in result.get(section, []):
                enforce_collection_status(acc)

        # Check that GPT returned all parser-detected inquiries
        from backend.core.logic.utils.names_normalization import normalize_bureau_name

        found_pairs = {
            (
                normalize_creditor_name(i.get("creditor_name")),
                i.get("date"),
                normalize_bureau_name(i.get("bureau", "")),
            )
            for i in result.get("inquiries", [])
        }
        for parsed in parsed_inquiries:
            key = (
                normalize_creditor_name(parsed["creditor_name"]),
                parsed["date"],
                normalize_bureau_name(parsed["bureau"]),
            )
            if key not in found_pairs:
                print(
                    f"[WARN] Inquiry missing from GPT output: {parsed['creditor_name']} {parsed['date']} ({parsed['bureau']})"
                )

    except Exception as e:
        print(f"[WARN] Late history parsing failed: {e}")

    issues = validate_analysis_sanity(result)
    if not result.get("open_accounts_with_issues") and detected_late_phrases(text):
        msg = (
            "WARN Late payment terms found in text but no accounts marked with issues."
        )
        issues.append(msg)
        print(msg)

    Path(output_json_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)

    return result
