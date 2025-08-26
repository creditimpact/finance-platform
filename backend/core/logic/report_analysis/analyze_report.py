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
import logging
import os
import re
from pathlib import Path
from typing import Any, Mapping

from rapidfuzz import fuzz

from backend.core.logic.report_analysis.candidate_logger import CandidateTokenLogger
from backend.core.logic.report_analysis.problem_detection import (
    evaluate_account_problem,
)
from backend.core.logic.utils.inquiries import extract_inquiries
from backend.core.logic.utils.names_normalization import normalize_creditor_name
from backend.core.logic.utils.norm import normalize_heading
from backend.core.logic.utils.text_parsing import (
    enforce_collection_status,
    extract_account_headings,
    extract_late_history_blocks,
)
from backend.core.services.ai_client import AIClient, get_ai_client

from .report_parsing import (
    extract_account_numbers,
    extract_creditor_remarks,
    extract_payment_statuses,
    extract_pdf_page_texts,
    extract_text_from_pdf,
    extract_three_column_fields,
    scan_page_markers,
)
from .report_postprocessing import (
    _assign_issue_types,
    _cleanup_unverified_late_text,
    _inject_missing_late_accounts,
    _merge_parser_inquiries,
    _reconcile_account_headings,
    _sanitize_late_counts,
    enrich_account_metadata,
    validate_analysis_sanity,
)
from .report_prompting import (
    ANALYSIS_PROMPT_VERSION,
    ANALYSIS_SCHEMA_VERSION,
    PIPELINE_VERSION,
    call_ai_analysis,
)

logger = logging.getLogger(__name__)


def _split_account_buckets(accounts: list[dict]) -> tuple[list[dict], list[dict]]:
    """Split accounts into negative and open issue buckets.

    The heuristics consider any charge-off/collection/closed indicators from
    merged fields such as ``payment_status`` and ``remarks``. Accounts that are
    currently open (e.g. "Open", "Current", "Pays as agreed") are classified
    under ``open_accounts_with_issues`` while all others default to
    ``negative_accounts``.
    """

    negatives: list[dict] = []
    open_issues: list[dict] = []
    for acc in accounts or []:
        if not acc.get("issue_types") and not acc.get("high_utilization"):
            continue

        parts = [
            acc.get("status"),
            acc.get("account_status"),
            acc.get("payment_status"),
            acc.get("remarks"),
        ]
        parts.extend((acc.get("payment_statuses") or {}).values())
        parts.extend((acc.get("status_texts") or {}).values())
        for fields in (acc.get("bureau_details") or {}).values():
            val = fields.get("account_status")
            if val:
                parts.append(val)
        status_text = " ".join(str(p) for p in parts if p).lower()

        evidence = {
            "status_text": status_text,
            "closed_date": acc.get("closed_date"),
            "past_due_amount": acc.get("past_due_amount"),
            "late_payments": bool(acc.get("late_payments")),
            "high_utilization": acc.get("high_utilization"),
        }

        negative_re = r"charge\s*off|charged\s*off|chargeoff|collection|derog|repossess"
        has_negative = bool(re.search(negative_re, status_text))
        if not has_negative and acc.get("closed_date"):
            has_negative = bool(
                re.search(r"derog|delinquent|charge|collection|repossess", status_text)
            )

        if has_negative and acc.get("primary_issue") == "late_payment":
            if "collection" in status_text:
                acc["primary_issue"] = "collection"
            elif re.search(r"charge\s*off|chargeoff", status_text):
                acc["primary_issue"] = "charge_off"
            elif "repossess" in status_text:
                acc["primary_issue"] = "repossessed"
            else:
                acc["primary_issue"] = "derogatory"
            acc.setdefault("issue_types", [])
            if acc["primary_issue"] not in acc["issue_types"]:
                acc["issue_types"].insert(0, acc["primary_issue"])

        if has_negative:
            bucket = "negative"
            negatives.append(acc)
        else:
            has_open = bool(
                re.search(r"\bopen\b|current|pays\s+as\s+agreed", status_text)
            ) and not acc.get("closed_date")
            has_issue = (
                (
                    isinstance(acc.get("past_due_amount"), (int, float))
                    and acc.get("past_due_amount") > 0
                )
                or bool(acc.get("late_payments"))
                or bool(acc.get("high_utilization"))
            )

            if has_open and has_issue:
                bucket = "open_issues"
                open_issues.append(acc)
            else:
                bucket = "negative"
                negatives.append(acc)

        logger.debug(
            "bucket_decision %s",
            json.dumps(
                {"name": acc.get("name"), "bucket": bucket, "evidence": evidence}
            ),
        )

    return negatives, open_issues


def _attach_parser_signals(
    accounts: list[dict] | None,
    payment_statuses_by_heading: dict[str, dict[str, str]],
    remarks_by_heading: dict[str, str],
    payment_status_raw_by_heading: dict[str, str],
) -> None:
    """Populate parser-derived fields for aggregated accounts."""

    for acc in accounts or []:
        if acc.get("source_stage") != "parser_aggregated":
            continue
        norm = acc.get("normalized_name") or normalize_creditor_name(
            acc.get("name", "")
        )
        acc["normalized_name"] = norm
        bureau_map = payment_statuses_by_heading.get(norm, {})
        acc["payment_statuses"] = bureau_map
        if bureau_map:
            acc["payment_status"] = "; ".join(sorted(set(bureau_map.values())))
        else:
            raw = payment_status_raw_by_heading.get(norm, "")
            acc["payment_status_raw"] = raw
            if raw:
                if re.search(r"\bcharge[-\s]?off\b", raw, re.I):
                    acc["payment_status"] = "charge_off"
                elif re.search(r"\bcollection(s)?\b", raw, re.I):
                    acc["payment_status"] = "collection"
        acc["remarks"] = remarks_by_heading.get(norm, "")


# ---------------------------------------------------------------------------
# Join helpers
# ---------------------------------------------------------------------------


def _fuzzy_match(name: str, choices: set[str]) -> str | None:
    """Return best fuzzy match for *name* within *choices* when >= 0.9."""

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


def _normalize_keys(mapping: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of *mapping* with creditor names normalized."""

    normalized: dict[str, Any] = {}
    for key, value in mapping.items():
        norm = normalize_creditor_name(key)
        existing = normalized.get(norm)
        if existing and isinstance(existing, dict) and isinstance(value, dict):
            existing.update(value)
        else:
            normalized[norm] = value
    return normalized


def _join_heading_map(
    accounts: Mapping[str, list[dict]],
    existing_norms: set[str],
    mapping: dict[str, Any],
    field_name: str | None,
    heading_map: Mapping[str, str],
    *,
    is_bureau_map: bool = False,
    aggregate_field: str | None = None,
) -> None:
    """Join a heading-keyed *mapping* onto *accounts* in-place.

    When *field_name* is ``None`` the mapping keys are reconciled (alias/fuzzy)
    but no fields are attached to accounts.  ``is_bureau_map`` controls whether
    values are ``{bureau: value}`` mappings that should be merged into the
    account's ``bureaus`` list.  For payment status maps an additional
    ``aggregate_field`` (e.g. ``"payment_status"``) may be specified to store a
    combined string value.
    """

    for key, value in list(mapping.items()):
        norm = normalize_creditor_name(key)
        raw = heading_map.get(norm, key)
        if norm != key:
            mapping.pop(key)
            if norm in mapping and field_name:
                if (
                    is_bureau_map
                    and isinstance(mapping[norm], dict)
                    and isinstance(value, dict)
                ):
                    mapping[norm].update(value)
                else:
                    mapping[norm] = value
            else:
                mapping[norm] = value
            value = mapping[norm]
        targets = accounts.get(norm)
        method: str | None = None
        if targets is None:
            match = _fuzzy_match(norm, existing_norms)
            if match:
                mapping.pop(norm)
                if match in mapping and field_name:
                    # merge dictionaries if both present
                    if (
                        is_bureau_map
                        and isinstance(mapping[match], dict)
                        and isinstance(value, dict)
                    ):
                        mapping[match].update(value)
                    else:
                        mapping[match] = value
                else:
                    mapping[match] = value
                targets = accounts.get(match)
                norm = match
                value = mapping[match]
                method = "fuzzy"
            else:
                details = {
                    "raw_key": raw,
                    "normalized": norm,
                    "target_present": False,
                    "map": field_name or "",
                }
                logger.debug(
                    "heading_join_miss %s",
                    json.dumps(details, sort_keys=True),
                )
                logger.info(
                    "heading_join_unresolved %s",
                    json.dumps(details, sort_keys=True),
                )
                continue
        elif raw.upper() != norm.upper():
            method = "alias"

        if field_name is None:
            if method:
                logger.info(
                    "heading_join_linked %s",
                    json.dumps(
                        {"raw_key": raw, "normalized_target": norm, "method": method},
                        sort_keys=True,
                    ),
                )
            continue

        for acc in targets or []:
            if is_bureau_map and isinstance(value, dict):
                acc.setdefault(field_name, {})
                acc[field_name].update(value)
                if aggregate_field:
                    acc[aggregate_field] = "; ".join(
                        sorted(set(acc[field_name].values()))
                    )
                acc.setdefault("bureaus", [])
                for bureau, val in value.items():
                    info = None
                    for b in acc["bureaus"]:
                        if isinstance(b, dict) and b.get("bureau") == bureau:
                            info = b
                            break
                    if info is None:
                        info = {"bureau": bureau}
                        acc["bureaus"].append(info)
                    if field_name == "payment_statuses":
                        if not info.get("payment_status"):
                            info["payment_status"] = val
                    else:
                        if not info.get(field_name):
                            info[field_name] = val
            else:
                acc[field_name] = value

        if method:
            logger.info(
                "heading_join_linked %s",
                json.dumps(
                    {"raw_key": raw, "normalized_target": norm, "method": method},
                    sort_keys=True,
                ),
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
    if os.getenv("EXPORT_RAW_PAGES", "0") != "0":
        pages = extract_pdf_page_texts(pdf_path)
        trace_dir = Path("trace") / request_id
        trace_dir.mkdir(parents=True, exist_ok=True)
        for idx, page in enumerate(pages, start=1):
            (trace_dir / f"page-{idx:02d}.txt").write_text(page, encoding="utf-8")
        markers = scan_page_markers(pages)
        logger.info("text_markers %s", json.dumps(markers, sort_keys=True))
        if not any(
            (
                markers["has_payment_status"],
                markers["has_creditor_remarks"],
                markers["has_account_status"],
            )
        ):
            logger.info('extraction_gap="no_marker_strings_found"')
    if not text.strip():
        raise ValueError("[ERROR] No text extracted from PDF")

    headings = extract_account_headings(text)
    heading_map = {normalize_creditor_name(norm): raw for norm, raw in headings}

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
        f"{text}|{ANALYSIS_PROMPT_VERSION}|{ANALYSIS_SCHEMA_VERSION}|{PIPELINE_VERSION}".encode(
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
            "needs_human_review": False,
            "missing_bureaus": [],
        }

    result["prompt_version"] = ANALYSIS_PROMPT_VERSION
    result["schema_version"] = ANALYSIS_SCHEMA_VERSION

    _reconcile_account_headings(result, heading_map)

    parsed_inquiries = extract_inquiries(text)
    inquiry_raw_map = {
        normalize_heading(i["creditor_name"]): i["creditor_name"]
        for i in parsed_inquiries
    }
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

    payment_status_map: dict[str, dict[str, str]] = {}
    _payment_status_raw_map: dict[str, str] = {}
    remarks_map: dict[str, dict[str, str]] = {}
    status_text_map: dict[str, dict[str, str]] = {}
    account_number_map: dict[str, dict[str, str]] = {}
    try:
        account_names = {acc.get("name", "") for acc in result.get("all_accounts", [])}
        history_all, raw_map, grid_all = extract_late_history_blocks(
            text, return_raw_map=True
        )
        _sanitize_late_counts(history_all)
        history_all = _normalize_keys(history_all)
        raw_map = _normalize_keys(raw_map)
        grid_all = _normalize_keys(grid_all)
        history, _, grid_map = extract_late_history_blocks(
            text, account_names, return_raw_map=True
        )
        _sanitize_late_counts(history)
        history = _normalize_keys(history)
        grid_map = _normalize_keys(grid_map)
        (
            col_payment_map,
            col_remarks_map,
            status_text_map,
            col_payment_raw,
            _col_remarks_raw,
            _col_status_raw,
            detail_map,
        ) = extract_three_column_fields(pdf_path)
        payment_status_map, _payment_status_raw_map = extract_payment_statuses(text)
        payment_status_map = _normalize_keys(payment_status_map)
        _payment_status_raw_map = _normalize_keys(_payment_status_raw_map)
        for name, vals in col_payment_map.items():
            norm = normalize_creditor_name(name)
            payment_status_map.setdefault(norm, {}).update(vals)
        for name, raw in col_payment_raw.items():
            norm = normalize_creditor_name(name)
            _payment_status_raw_map.setdefault(norm, raw)
        remarks_map = extract_creditor_remarks(text)
        remarks_map = _normalize_keys(remarks_map)
        for name, vals in col_remarks_map.items():
            norm = normalize_creditor_name(name)
            remarks_map.setdefault(norm, {}).update(vals)
        account_number_map = extract_account_numbers(text)
        account_number_map = _normalize_keys(account_number_map)
        for acc_name, bureaus in detail_map.items():
            acc_norm = normalize_creditor_name(acc_name)
            for bureau, fields in bureaus.items():
                num = fields.get("account_number")
                if num:
                    account_number_map.setdefault(acc_norm, {})[bureau] = str(num)
        status_text_map = _normalize_keys(status_text_map)
        detail_map = _normalize_keys(detail_map)

        if history:
            print(f"[INFO] Found {len(history)} late payment block(s):")
            for creditor, bureaus in history.items():
                print(
                    f"[INFO] Detected late payments for: '{creditor.title()}' -> {bureaus}"
                )
        else:
            print("[ERROR] No late payment history blocks detected.")

        accounts_by_norm: dict[str, list[dict]] = {}
        sections = [
            "all_accounts",
            "negative_accounts",
            "open_accounts_with_issues",
            "positive_accounts",
            "high_utilization_accounts",
        ]
        for section in sections:
            for acc in result.get(section, []):
                raw_name = acc.get("name", "")
                norm = normalize_creditor_name(raw_name)
                if raw_name and norm != raw_name.lower().strip():
                    print(f"[~] Normalized account heading '{raw_name}' -> '{norm}'")
                acc["normalized_name"] = norm
                accounts_by_norm.setdefault(norm, []).append(acc)
        existing_norms = set(accounts_by_norm.keys())

        _join_heading_map(
            accounts_by_norm, existing_norms, history, "late_payments", raw_map
        )
        _join_heading_map(
            accounts_by_norm, existing_norms, grid_map, "grid_history_raw", raw_map
        )
        _join_heading_map(
            accounts_by_norm,
            existing_norms,
            payment_status_map,
            "payment_statuses",
            heading_map,
            is_bureau_map=True,
            aggregate_field="payment_status",
        )
        _join_heading_map(
            accounts_by_norm,
            existing_norms,
            _payment_status_raw_map,
            "payment_status_raw",
            heading_map,
        )
        _join_heading_map(
            accounts_by_norm,
            existing_norms,
            remarks_map,
            "remarks",
            heading_map,
            is_bureau_map=True,
        )
        _join_heading_map(
            accounts_by_norm,
            existing_norms,
            status_text_map,
            "account_status",
            heading_map,
            is_bureau_map=True,
        )
        _join_heading_map(
            accounts_by_norm,
            existing_norms,
            account_number_map,
            None,
            heading_map,
        )
        # Reconcile global maps for later bookkeeping
        _join_heading_map(accounts_by_norm, existing_norms, history_all, None, raw_map)
        _join_heading_map(accounts_by_norm, existing_norms, grid_all, None, raw_map)

        def _apply_late_flags(acc_list, section_name):
            for acc in acc_list or []:
                norm = acc.get("normalized_name")
                if norm in history and any(
                    v >= 1 for vals in history[norm].values() for v in vals.values()
                ):
                    acc.setdefault("flags", []).append("Late Payments")
                    if section_name not in [
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

        for section in sections:
            _apply_late_flags(result.get(section, []), section)

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
                norm = acc.get("normalized_name") or normalize_creditor_name(
                    acc.get("name", "")
                )
                acc["normalized_name"] = norm
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

        _inject_missing_late_accounts(result, history_all, raw_map, grid_all)

        _attach_parser_signals(
            result.get("all_accounts"),
            payment_status_map,
            remarks_map,
            _payment_status_raw_map,
        )

        _merge_parser_inquiries(result, parsed_inquiries, inquiry_raw_map)

        def _merge_account_numbers(acc_list, field_map):
            for acc in acc_list or []:
                norm = acc.get("normalized_name") or normalize_creditor_name(
                    acc.get("name", "")
                )
                raw_name = acc.get("name", "")
                acc["normalized_name"] = norm
                values_map = field_map.get(norm)
                if not values_map:
                    logger.info(
                        "heading_join_unresolved %s",
                        json.dumps(
                            {
                                "raw_key": raw_name,
                                "normalized": norm,
                                "target_present": False,
                                "map": "account_number",
                            },
                            sort_keys=True,
                        ),
                    )
                    continue
                logger.info(
                    "heading_join_linked %s",
                    json.dumps(
                        {
                            "raw_key": raw_name,
                            "normalized_target": norm,
                            "method": "canonical",
                        },
                        sort_keys=True,
                    ),
                )
                acc.setdefault("bureaus", [])
                raw_unique: set[str] = set()
                digit_unique: set[str] = set()
                for bureau, raw in values_map.items():
                    info = None
                    for b in acc["bureaus"]:
                        if isinstance(b, dict) and b.get("bureau") == bureau:
                            info = b
                            break
                    if info is None:
                        info = {"bureau": bureau}
                        acc["bureaus"].append(info)
                    if not info.get("account_number_raw"):
                        info["account_number_raw"] = raw
                    digits = re.sub(r"\D", "", raw)
                    if digits and not info.get("account_number"):
                        info["account_number"] = digits
                    if raw:
                        raw_unique.add(raw)
                    if digits:
                        digit_unique.add(digits)
                if not acc.get("account_number_raw") and len(raw_unique) == 1:
                    acc["account_number_raw"] = next(iter(raw_unique))
                if not acc.get("account_number") and len(digit_unique) == 1:
                    acc["account_number"] = next(iter(digit_unique))

        def _merge_bureau_details(acc_list, detail_map):
            for acc in acc_list or []:
                norm = acc.get("normalized_name") or normalize_creditor_name(
                    acc.get("name", "")
                )
                raw_name = acc.get("name", "")
                acc["normalized_name"] = norm
                values_map = detail_map.get(norm)
                if not values_map:
                    logger.info(
                        "heading_join_unresolved %s",
                        json.dumps(
                            {
                                "raw_key": raw_name,
                                "normalized": norm,
                                "target_present": False,
                                "map": "bureau_details",
                            },
                            sort_keys=True,
                        ),
                    )
                    continue
                logger.info(
                    "heading_join_linked %s",
                    json.dumps(
                        {
                            "raw_key": raw_name,
                            "normalized_target": norm,
                            "method": "canonical",
                        },
                        sort_keys=True,
                    ),
                )
                bd = acc.setdefault("bureau_details", {})
                for bureau, fields in values_map.items():
                    bd.setdefault(bureau, {}).update(fields)

        for sec in [
            "all_accounts",
            "negative_accounts",
            "open_accounts_with_issues",
            "positive_accounts",
            "high_utilization_accounts",
        ]:
            _merge_account_numbers(result.get(sec, []), account_number_map)
            _merge_bureau_details(result.get(sec, []), detail_map)

        for section in [
            "all_accounts",
            "negative_accounts",
            "open_accounts_with_issues",
            "positive_accounts",
            "high_utilization_accounts",
        ]:
            for acc in result.get(section, []):
                enforce_collection_status(acc)

        candidate_logger = CandidateTokenLogger()
        for acc in result.get("all_accounts", []):
            candidate_logger.collect(acc)
            verdict = evaluate_account_problem(acc)
            acc["primary_issue"] = verdict["primary_issue"]
            acc["problem_reasons"] = verdict["problem_reasons"]
            acc["decision_source"] = verdict["decision_source"]
            acc["confidence"] = verdict["confidence"]
            acc["supporting"] = verdict["supporting"]
            acc["_detector_is_problem"] = verdict["is_problem"]
        candidate_logger.save(Path("client_output") / request_id)

        result["problem_accounts"] = [
            a for a in result.get("all_accounts", []) if a.get("_detector_is_problem")
        ]

        if os.getenv("DEFER_ASSIGN_ISSUE_TYPES") == "1":
            for acc in result.get("all_accounts", []):
                acc["primary_issue"] = "unknown"
                acc["issue_types"] = []
            result["negative_accounts"] = []
            result["open_accounts_with_issues"] = []
            result["positive_accounts"] = []
            result["high_utilization_accounts"] = []
        else:
            for acc in result.get("all_accounts", []):
                _assign_issue_types(acc)

            negatives, open_issues = _split_account_buckets(
                result.get("all_accounts", [])
            )
            result["negative_accounts"] = negatives
            result["open_accounts_with_issues"] = open_issues

        # Check that GPT returned all parser-detected inquiries
        from backend.core.logic.utils.names_normalization import normalize_bureau_name

        found_pairs = {
            (
                normalize_heading(i.get("creditor_name")),
                i.get("date"),
                normalize_bureau_name(i.get("bureau", "")),
            )
            for i in result.get("inquiries", [])
        }
        for parsed in parsed_inquiries:
            key = (
                normalize_heading(parsed["creditor_name"]),
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

    for section in [
        "all_accounts",
        "negative_accounts",
        "open_accounts_with_issues",
        "positive_accounts",
        "high_utilization_accounts",
    ]:
        result[section] = [
            enrich_account_metadata(acc) for acc in result.get(section, [])
        ]

    Path(output_json_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)

    return result
