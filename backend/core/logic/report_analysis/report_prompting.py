"""Prompt construction and AI calls for credit report analysis."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from copy import deepcopy
from pathlib import Path
from typing import Dict, List

from jsonschema import Draft7Validator

from backend.analytics.analytics_tracker import log_ai_request
from backend.audit.audit import emit_event
from backend.core.logic.report_analysis.flags import FLAGS
from backend.core.logic.utils.inquiries import extract_inquiries
from backend.core.logic.utils.json_utils import parse_json
from backend.core.logic.utils.names_normalization import (
    BUREAUS,
    normalize_bureau_name,
    normalize_creditor_name,
)
from backend.core.logic.utils.pii import redact_pii
from backend.core.logic.utils.text_parsing import (
    extract_account_headings,
    extract_late_history_blocks,
)
from backend.core.services.ai_client import AIClient

from .analysis_cache import get_cached_analysis, store_cached_analysis

_INPUT_COST_PER_TOKEN = 0.01 / 1000
_OUTPUT_COST_PER_TOKEN = 0.03 / 1000
ANALYSIS_MODEL_VERSION = "gpt-4-turbo"

_SCHEMA_PATH = Path(__file__).with_name("analysis_schema.json")
_ANALYSIS_SCHEMA = json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))
_ANALYSIS_VALIDATOR = Draft7Validator(_ANALYSIS_SCHEMA)
# ANALYSIS_PROMPT_VERSION history:
# 2: Add explicit JSON directive (Task 8)
# 1: Initial version
ANALYSIS_PROMPT_VERSION = 2
ANALYSIS_SCHEMA_VERSION = 1


def _apply_defaults(data: dict, schema: dict) -> None:
    """Recursively populate defaults based on ``schema``."""
    for key, subschema in schema.get("properties", {}).items():
        if key not in data:
            if "default" in subschema:
                data[key] = deepcopy(subschema["default"])
            else:
                t = subschema.get("type")
                if t == "array":
                    data[key] = []
                elif t in {"number", "integer"}:
                    data[key] = 0
                elif t == "boolean":
                    data[key] = False
                elif t == "object":
                    data[key] = {}
            if subschema.get("type") == "object" and isinstance(data[key], dict):
                _apply_defaults(data[key], subschema)
        elif subschema.get("type") == "object" and isinstance(data[key], dict):
            _apply_defaults(data[key], subschema)


def _validate_analysis_schema(data: dict) -> dict:
    """Validate ``data`` against the analysis schema and fill defaults."""
    errors = [e.message for e in _ANALYSIS_VALIDATOR.iter_errors(data)]
    if errors:
        logging.warning(
            "report_analysis_schema_validation_failed",
            extra={"validation_errors": errors},
        )
    _apply_defaults(data, _ANALYSIS_SCHEMA)
    return data


def _split_text_by_bureau(text: str) -> Dict[str, str]:
    """Return mapping of bureau name to its text segment."""
    positions: Dict[str, int] = {}
    for bureau in BUREAUS:
        match = re.search(bureau, text, re.I)
        if match:
            positions[bureau] = match.start()
    if not positions:
        return {"Full": text}
    sorted_pos = sorted(positions.items(), key=lambda x: x[1])
    segments: Dict[str, str] = {}
    for i, (bureau, start) in enumerate(sorted_pos):
        end = sorted_pos[i + 1][1] if i + 1 < len(sorted_pos) else len(text)
        segments[bureau] = text[start:end]
    return segments


def _generate_prompt(
    segment_text: str,
    *,
    is_identity_theft: bool,
    strategic_context: str | None,
) -> tuple[str, str, str]:
    """Return prompt text and summaries used for a segment."""
    if FLAGS.inject_headings:
        headings = extract_account_headings(segment_text)
        if headings:
            heading_summary = "Account Headings:\n" + "\n".join(
                f"- {norm.title()} (raw: {raw})" for norm, raw in headings
            )
        else:
            heading_summary = "Account Headings:\n- None detected"
    else:
        heading_summary = ""

    late_blocks, late_raw_map = extract_late_history_blocks(
        segment_text, return_raw_map=True
    )
    if late_blocks:
        late_summary_text = "Late payment history extracted from report:\n"
        for account, bureaus in late_blocks.items():
            raw = late_raw_map.get(account, account)
            for bureau, counts in bureaus.items():
                parts = [f"{k} days late: {v}" for k, v in counts.items()]
                goodwill = False
                thirty = counts.get("30", 0)
                sixty = counts.get("60", 0)
                ninety = counts.get("90", 0)
                if (thirty == 1 and not sixty and not ninety) or (
                    thirty == 2 and not sixty and not ninety
                ):
                    goodwill = True
                line = f"- {account} (raw: {raw}) ({bureau}): {', '.join(parts)}"
                if goodwill:
                    line += " \u2192 Possible goodwill candidate"
                late_summary_text += line + "\n"
    else:
        late_summary_text = (
            "No late payment blocks were detected by parsing logic. "
            "Do not infer or guess late payment history unless clearly shown."
        )

    inquiry_list = extract_inquiries(segment_text)
    if inquiry_list:
        inquiry_summary = "\nParsed inquiries from report:\n" + "\n".join(
            f"- {normalize_creditor_name(i['creditor_name'])} "
            f"(raw: {i['creditor_name']}) {i['date']} ({i['bureau']})"
            for i in inquiry_list
        )
    else:
        inquiry_summary = "\nNo inquiries were detected by the parser."

    strategic_context_line = (
        f'Strategic context: "{strategic_context}"\n' if strategic_context else ""
    )

    prompt = f"""{heading_summary}

Respond with **valid JSON only**; use `null` or empty arrays when data is unknown; do **not** invent fields.

You are a senior credit repair expert with deep knowledge of credit reports, FCRA regulations, dispute strategies, and client psychology.

Your task: deeply analyze the following SmartCredit report text and extract a structured JSON summary for use in automated dispute and goodwill letters, and personalized client instructions.

Identity theft case: {"Yes" if is_identity_theft else "No"}
{strategic_context_line}Return only valid JSON with all property names in double quotes. No comments or extra text outside the JSON object.

Return this exact JSON structure:

1. personal_info_issues: List of inconsistencies or mismatches in personal info (name, address, DOB) across bureaus.

2. negative_accounts: Accounts marked as Chargeoff or in Collections.
   For each account, include:
   - name
   - balance (e.g., "$2,356")
   - status ("Chargeoff" or "Collections")
   - account_status ("Open" or "Closed")
   - account_number (if visible)
   - bureaus (["Experian", "Equifax", "TransUnion"])
   - is_suspected_identity_theft: true/false
   - dispute_type: "identity_theft" / "unauthorized_or_unverified" / "inaccurate_reporting"
   - impact: short sentence describing how this negatively affects the client
   - advisor_comment: 1–2 sentences explaining what we’re doing about this account and why.

3. open_accounts_with_issues: Non-collection accounts with any late payment history or remarks like "past due", "derogatory", etc. Look carefully for text such as "30 days late", "60 days late", or "past due" even if formatting is odd. If an account is otherwise in good standing with only one or two late payments, mark ``goodwill_candidate`` true.
   Include:
   - name
   - status
   - account_status
   - account_number (if available)
   - bureaus
   - goodwill_candidate: true/false
   - is_suspected_identity_theft: true/false
   - dispute_type
   - hardship_reason: (e.g. "temporary job loss")
   - recovery_summary
   - advisor_comment

4. high_utilization_accounts: Revolving accounts with balance over 30% of limit.
   Include:

{late_summary_text}

{inquiry_summary}

Report text:
===
{segment_text}
===
"""

    return prompt, late_summary_text, inquiry_summary


def _run_segment(
    segment_text: str,
    *,
    is_identity_theft: bool,
    output_json_path: Path,
    ai_client: AIClient,
    strategic_context: str | None,
    prompt: str,
    late_summary_text: str,
    inquiry_summary: str,
) -> tuple[dict, str | None]:
    """Run the existing prompt/analysis flow for a single bureau segment.

    Returns ``(data, error_code)`` where ``error_code`` is ``None`` on success or
    one of ``BROKEN_JSON``, ``TIMEOUT``, ``EMPTY_OUTPUT`` for known failure
    scenarios. Any unexpected exception will be logged and returned as its class
    name.
    """
    try:
        response = ai_client.chat_completion(
            model=ANALYSIS_MODEL_VERSION,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )
    except TimeoutError:
        _validate_analysis_schema({})  # ensure validation errors get logged
        return {}, "TIMEOUT"
    except Exception as exc:  # pragma: no cover - defensive
        logging.exception("segment_call_failed", exc_info=exc)
        _validate_analysis_schema({})
        return {}, type(exc).__name__

    content = response.choices[0].message.content.strip()

    if content.startswith("```"):
        content = content.replace("```json", "").replace("```", "").strip()

    raw_path = output_json_path.with_name(output_json_path.stem + "_raw.txt")
    if FLAGS.debug_store_raw:
        red_content = redact_pii(content)
        red_late = redact_pii(late_summary_text)
        red_inquiry = redact_pii(inquiry_summary)
        with open(raw_path, "w", encoding="utf-8") as f:
            f.write(red_content)
            f.write("\n\n---\n[Debug] Late Payment Summary Used in Prompt:\n")
            f.write(red_late)
            f.write("\n\n---\n[Debug] Inquiry Summary Used in Prompt:\n")
            f.write(red_inquiry)

    if not content:
        _validate_analysis_schema({})
        return {}, "EMPTY_OUTPUT"

    data, parse_error = parse_json(content)
    if parse_error:
        data = _validate_analysis_schema({})
        return data, "BROKEN_JSON"

    data = _validate_analysis_schema(data)
    return data, None


def _merge_accounts(dest: List[dict], new: List[dict]) -> None:
    """Merge account lists, combining bureau info."""
    index = {
        (normalize_creditor_name(a.get("name", "")), a.get("account_number")): a
        for a in dest
    }
    for acc in new:
        key = (
            normalize_creditor_name(acc.get("name", "")),
            acc.get("account_number"),
        )
        existing = index.get(key)
        if existing:
            existing_bureaus = {
                normalize_bureau_name(b) for b in existing.get("bureaus", [])
            }
            new_bureaus = {normalize_bureau_name(b) for b in acc.get("bureaus", [])}
            existing["bureaus"] = sorted(existing_bureaus | new_bureaus)
            for k, v in acc.items():
                if k == "bureaus":
                    continue
                if k not in existing or not existing[k]:
                    existing[k] = v
        else:
            acc["bureaus"] = [normalize_bureau_name(b) for b in acc.get("bureaus", [])]
            dest.append(acc)
            index[key] = acc


def _merge_inquiries(dest: List[dict], new: List[dict]) -> None:
    """Merge inquiry lists without duplicates."""
    index = {
        (
            normalize_creditor_name(i.get("creditor_name")),
            i.get("date"),
            normalize_bureau_name(i.get("bureau")),
        ): i
        for i in dest
    }
    for inq in new:
        key = (
            normalize_creditor_name(inq.get("creditor_name")),
            inq.get("date"),
            normalize_bureau_name(inq.get("bureau")),
        )
        if key not in index:
            inq["bureau"] = normalize_bureau_name(inq.get("bureau"))
            dest.append(inq)
            index[key] = inq


def call_ai_analysis(
    text: str,
    is_identity_theft: bool,
    output_json_path: Path,
    ai_client: AIClient,
    strategic_context: str | None = None,
    *,
    request_id: str,
    doc_fingerprint: str,
):
    """Analyze raw report text using an AI model and return parsed JSON."""
    logging.debug("analysis_flags", extra={"flags": FLAGS.__dict__})
    segments = _split_text_by_bureau(text) if FLAGS.chunk_by_bureau else {"Full": text}

    aggregate: dict = {
        "negative_accounts": [],
        "open_accounts_with_issues": [],
        "positive_accounts": [],
        "high_utilization_accounts": [],
        "all_accounts": [],
        "inquiries": [],
        "personal_info_issues": [],
        "account_inquiry_matches": [],
        "strategic_recommendations": [],
    }
    summary_metrics: dict = {}

    for idx, (bureau, seg_text) in enumerate(segments.items()):
        seg_path = (
            output_json_path
            if idx == 0
            else output_json_path.with_name(f"{output_json_path.stem}_{bureau}.json")
        )
        start = time.time()
        data: dict = {}
        error_code: str | None = None
        attempt = 0
        seg_text_attempt = seg_text
        tokens_in = tokens_out = 0
        while attempt < 3:
            attempt += 1
            try:
                prompt, late_summary_text, inquiry_summary = _generate_prompt(
                    seg_text_attempt,
                    is_identity_theft=is_identity_theft,
                    strategic_context=strategic_context,
                )
                prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
                cache_entry = get_cached_analysis(
                    doc_fingerprint,
                    bureau,
                    prompt_hash,
                    ANALYSIS_MODEL_VERSION,
                    prompt_version=ANALYSIS_PROMPT_VERSION,
                    schema_version=ANALYSIS_SCHEMA_VERSION,
                )
                if cache_entry is not None:
                    data = cache_entry
                    error_code = None
                else:
                    data, error_code = _run_segment(
                        seg_text_attempt,
                        is_identity_theft=is_identity_theft,
                        output_json_path=seg_path,
                        ai_client=ai_client,
                        strategic_context=strategic_context,
                        prompt=prompt,
                        late_summary_text=late_summary_text,
                        inquiry_summary=inquiry_summary,
                    )
                    headings = extract_account_headings(seg_text_attempt)
                    if headings:
                        account_names = set()
                        for list_name in [
                            "all_accounts",
                            "negative_accounts",
                            "open_accounts_with_issues",
                            "positive_accounts",
                            "high_utilization_accounts",
                        ]:
                            for acc in data.get(list_name, []):
                                account_names.add(
                                    normalize_creditor_name(acc.get("name", ""))
                                )
                        unmatched_norms = {
                            norm for norm, _ in headings if norm not in account_names
                        }
                        unmatched_raws = [
                            raw for norm, raw in headings if norm in unmatched_norms
                        ]
                        if unmatched_raws:
                            logging.info(
                                "negative_headings_without_accounts",
                                extra={"unmatched_headings": unmatched_raws},
                            )
                        if not error_code and headings:
                            match_rate = (
                                (len(headings) - len(unmatched_norms)) / len(headings)
                            ) * 100
                            if match_rate < 70:
                                logging.warning(
                                    "low_recall_accounts",
                                    extra={
                                        "match_rate": match_rate,
                                        "validation_errors": ["LOW_RECALL"],
                                    },
                                )
                                error_code = "LOW_RECALL"
                    if not error_code:
                        store_cached_analysis(
                            doc_fingerprint,
                            bureau,
                            prompt_hash,
                            ANALYSIS_MODEL_VERSION,
                            data,
                            prompt_version=ANALYSIS_PROMPT_VERSION,
                            schema_version=ANALYSIS_SCHEMA_VERSION,
                        )
            except Exception as exc:  # pragma: no cover - defensive
                error_code = getattr(exc, "code", type(exc).__name__)
                data = {}

            logging.info(
                "analysis_attempt",
                extra={
                    "bureau": bureau,
                    "attempt": attempt,
                    "error_code": error_code or 0,
                },
            )

            if not error_code:
                break
            if attempt < 3:
                time.sleep(0.5 * attempt)
                if error_code in {"BROKEN_JSON", "TIMEOUT", "EMPTY_OUTPUT"}:
                    seg_text_attempt = seg_text_attempt[
                        : max(50, len(seg_text_attempt) // 2)
                    ]

        logging.info(
            "analysis_final",
            extra={
                "bureau": bureau,
                "attempts": attempt,
                "final_error": error_code or 0,
            },
        )

        latency_ms = (time.time() - start) * 1000
        emit_event(
            "report_segment",
            {
                "request_id": request_id,
                "doc_fingerprint": doc_fingerprint,
                "bureau": bureau,
                "prompt_version": ANALYSIS_PROMPT_VERSION,
                "schema_version": ANALYSIS_SCHEMA_VERSION,
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
                "latency_ms": latency_ms,
                "error_code": error_code or 0,
                "attempts": attempt,
            },
        )
        cost = tokens_in * _INPUT_COST_PER_TOKEN + tokens_out * _OUTPUT_COST_PER_TOKEN
        log_ai_request(tokens_in, tokens_out, cost, latency_ms)

        for key in [
            "negative_accounts",
            "open_accounts_with_issues",
            "positive_accounts",
            "high_utilization_accounts",
            "all_accounts",
        ]:
            if data.get(key):
                _merge_accounts(aggregate[key], data[key])

        if data.get("inquiries"):
            _merge_inquiries(aggregate["inquiries"], data["inquiries"])

        if data.get("summary_metrics"):
            for k, v in data["summary_metrics"].items():
                if isinstance(v, bool):
                    summary_metrics[k] = summary_metrics.get(k, False) or v
                elif isinstance(v, list):
                    summary_metrics[k] = sorted(set(summary_metrics.get(k, []) + v))
                elif isinstance(v, (int, float)):
                    summary_metrics[k] = summary_metrics.get(k, 0) + v

        for key in [
            "personal_info_issues",
            "account_inquiry_matches",
            "strategic_recommendations",
        ]:
            if data.get(key):
                aggregate[key].extend(data[key])

    if summary_metrics:
        aggregate["summary_metrics"] = summary_metrics
    aggregate["prompt_version"] = ANALYSIS_PROMPT_VERSION
    aggregate["schema_version"] = ANALYSIS_SCHEMA_VERSION

    return aggregate
