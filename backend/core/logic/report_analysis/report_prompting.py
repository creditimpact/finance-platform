"""Prompt construction and AI calls for credit report analysis."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from copy import deepcopy
from datetime import datetime
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


# Allow for odd spacing, lowercase headers, and page-break markers when
# locating bureau sections in raw report text.
_BUREAU_REGEXES = {
    bureau: re.compile(
        r"(?:^|\n|\f)\s*" + r"[\s-]*".join(re.escape(ch) for ch in bureau) + r"\b",
        re.IGNORECASE,
    )
    for bureau in BUREAUS
}


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


# Example object matching the analysis schema used in prompts
_SCHEMA_EXAMPLE = json.dumps(_validate_analysis_schema({}), indent=2)

def _split_text_by_bureau(text: str) -> Dict[str, str]:
    """Return mapping of bureau name to its text segment."""
    positions: Dict[str, int] = {}
    for bureau, pattern in _BUREAU_REGEXES.items():
        match = pattern.search(text)
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


def log_bureau_failure(
    *,
    error_code: str,
    bureau: str,
    expected_headings: int,
    found_accounts: int,
    tokens: int,
    latency: float,
) -> None:
    """Emit structured failure log for alerting dashboards."""

    payload = {
        "error_code": error_code,
        "bureau": bureau,
        "expected_headings": expected_headings,
        "found_accounts": found_accounts,
        "tokens": tokens,
        "latency": latency,
    }
    emit_event("bureau_failure", payload)


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


def analyze_bureau(
    text: str,
    *,
    is_identity_theft: bool,
    output_json_path: Path,
    ai_client: AIClient,
    strategic_context: str | None,
    prompt: str,
    late_summary_text: str,
    inquiry_summary: str,
    hints: dict | None = None,
) -> tuple[dict, str | None]:
    """Run the prompt/analysis flow for a single bureau segment.

    Parameters are identical to the legacy ``_run_segment`` helper with the
    addition of ``hints`` which may contain optional keys such as
    ``expected_account_names`` or ``extra_instructions``. The function now
    returns ``(data, error_code)``.
    """
    tokens_in = tokens_out = 0
    hints = hints or {}
    expected_accounts = hints.get("expected_account_names") or []
    extra_instructions = hints.get("extra_instructions")
    if expected_accounts:
        prompt += "\n\nExpected account names:\n" + "\n".join(
            f"- {a}" for a in expected_accounts
        )
    if extra_instructions:
        prompt += "\n\n" + extra_instructions
    try:
        prompt_with_schema = (
            prompt + "\n\nJSON schema exemplar:\n" + _SCHEMA_EXAMPLE
        )
        response = ai_client.chat_completion(
            model=ANALYSIS_MODEL_VERSION,
            messages=[{"role": "user", "content": prompt_with_schema}],
            temperature=0.1,
        )
        usage = getattr(response, "usage", None)
        if usage:
            tokens_in = getattr(usage, "prompt_tokens", 0) or getattr(
                usage, "input_tokens", 0
            )
            tokens_out = getattr(usage, "completion_tokens", 0) or getattr(
                usage, "output_tokens", 0
            )
    except TimeoutError:
        data = _validate_analysis_schema({})  # ensure validation errors get logged
        data["confidence"] = 0.0
        return data, "TIMEOUT"
    except Exception as exc:  # pragma: no cover - defensive
        logging.exception("segment_call_failed", exc_info=exc)
        data = _validate_analysis_schema({})
        data["confidence"] = 0.0
        return data, type(exc).__name__

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
        data = _validate_analysis_schema({})
        data["confidence"] = 0.0
        return data, "EMPTY_OUTPUT"

    data, parse_error = parse_json(content)
    if parse_error:
        data = _validate_analysis_schema({})
        data["confidence"] = 0.0
        return data, "BROKEN_JSON"

    data = _validate_analysis_schema(data)
    validation_errors = list(_ANALYSIS_VALIDATOR.iter_errors(data))
    if validation_errors:
        data["confidence"] = 0.0
        return data, "SCHEMA_VALIDATION_FAILED"

    # ------------------------------------------------------------------
    # Basic confidence heuristic
    # ------------------------------------------------------------------
    headings = extract_account_headings(text)
    if expected_accounts:
        headings.extend(
            (normalize_creditor_name(a), a) for a in expected_accounts
        )
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
                account_names.add(normalize_creditor_name(acc.get("name", "")))
        unmatched_norms = {norm for norm, _ in headings if norm not in account_names}
        confidence = (len(headings) - len(unmatched_norms)) / len(headings)
    else:
        confidence = 1.0
    data["confidence"] = confidence

    for list_name in [
        "all_accounts",
        "negative_accounts",
        "open_accounts_with_issues",
        "positive_accounts",
        "high_utilization_accounts",
    ]:
        for acc in data.get(list_name, []):
            acc.setdefault("confidence", confidence)
    for inq in data.get("inquiries", []):
        inq.setdefault("confidence", confidence)

    return data, None


def _parse_date(date_str: str | None) -> datetime | None:
    """Parse a date string using common formats."""
    if not date_str:
        return None
    for fmt in ("%m/%d/%Y", "%m-%d-%Y", "%m.%d.%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(date_str, fmt)
        except Exception:
            continue
    return None


def _date_bucket(date_str: str | None) -> int | None:
    """Return a 7-day bucket identifier for ``date_str``."""
    dt = _parse_date(date_str)
    if not dt:
        return None
    return dt.toordinal() // 7


def _balance_bucket(balance: str | None) -> int | None:
    """Bucketize balance amounts to $100 increments."""
    if balance is None:
        return None
    clean = re.sub(r"[^0-9.]+", "", str(balance))
    if not clean:
        return None
    try:
        amount = float(clean)
    except Exception:
        return None
    return int(amount // 100)


def _merge_accounts(dest: List[dict], new: List[dict]) -> None:
    """Merge account lists using weighted keys."""

    def _account_keys(acc: dict, include_variants: bool) -> List[tuple]:
        name = normalize_creditor_name(acc.get("name", ""))
        bucket = _date_bucket(acc.get("opened_date"))
        bal_bucket = _balance_bucket(acc.get("balance"))
        bureaus = acc.get("bureaus") or (
            [acc.get("bureau")] if acc.get("bureau") else []
        )
        bureaus = [normalize_bureau_name(b) for b in bureaus]
        bureaus.append(None)
        buckets = [bucket]
        if include_variants and bucket is not None:
            buckets.extend([bucket - 1, bucket + 1])
        keys = []
        for b in bureaus:
            for db in buckets:
                keys.append((name, db, bal_bucket, b))
        return keys

    index: dict[tuple, dict] = {}
    for a in dest:
        bureaus = a.get("bureaus") or ([a.get("bureau")] if a.get("bureau") else [])
        a["bureaus"] = [normalize_bureau_name(b) for b in bureaus]
        for key in _account_keys(a, include_variants=False):
            index[key] = a

    for acc in new:
        bureaus = acc.get("bureaus") or (
            [acc.get("bureau")] if acc.get("bureau") else []
        )
        bureaus = [normalize_bureau_name(b) for b in bureaus]
        acc["bureaus"] = bureaus
        existing = None
        for key in _account_keys(acc, include_variants=True):
            if key in index:
                existing = index[key]
                break
        if existing:
            existing_bureaus = {
                normalize_bureau_name(b) for b in existing.get("bureaus", [])
            }
            existing["bureaus"] = sorted(existing_bureaus | set(bureaus))
            for k, v in acc.items():
                if k == "bureaus":
                    continue
                if k == "confidence":
                    existing["confidence"] = max(existing.get("confidence", 0), v)
                    continue
                if k not in existing or not existing[k]:
                    existing[k] = v
            for key in _account_keys(existing, include_variants=False):
                index[key] = existing
        else:
            dest.append(acc)
            for key in _account_keys(acc, include_variants=False):
                index[key] = acc


def _merge_inquiries(dest: List[dict], new: List[dict]) -> None:
    """Merge inquiry lists without duplicates using weighted keys."""

    def _inq_keys(inq: dict, include_variants: bool) -> List[tuple]:
        name = normalize_creditor_name(inq.get("creditor_name"))
        bucket = _date_bucket(inq.get("date"))
        bureau = normalize_bureau_name(inq.get("bureau"))
        buckets = [bucket]
        if include_variants and bucket is not None:
            buckets.extend([bucket - 1, bucket + 1])
        return [(name, b, 0, bureau) for b in buckets]

    index = {}
    for i in dest:
        for key in _inq_keys(i, include_variants=False):
            index[key] = i
    for inq in new:
        existing = None
        for key in _inq_keys(inq, include_variants=True):
            if key in index:
                existing = index[key]
                break
        if existing:
            existing["confidence"] = max(
                existing.get("confidence", 0), inq.get("confidence", 0)
            )
        else:
            inq["bureau"] = normalize_bureau_name(inq.get("bureau"))
            dest.append(inq)
            for key in _inq_keys(inq, include_variants=False):
                index[key] = inq


def _detect_segment_issues(
    seg_text: str, data: dict, bureau: str
) -> List[tuple[str, List[str]]]:
    """Return list of (issue_code, related_items) tuples for a segment.

    Issues detected:

    - ``MISSING_EXPECTED_ACCOUNT``: An account heading was parsed from
      ``seg_text`` but no corresponding account was returned by the model.
    - ``MERGED_ACCOUNTS``: The model returned an account referencing multiple
      bureaus, suggesting separate accounts were merged together.
    """

    issues: List[tuple[str, List[str]]] = []

    headings = extract_account_headings(seg_text)
    account_names = {
        normalize_creditor_name(a.get("name", "")) for a in data.get("all_accounts", [])
    }
    missing = [raw for norm, raw in headings if norm not in account_names]
    if missing:
        issues.append(("MISSING_EXPECTED_ACCOUNT", missing))

    merged = []
    for acc in data.get("all_accounts", []):
        bureaus = {normalize_bureau_name(b) for b in acc.get("bureaus", [])}
        if len(bureaus) > 1 or (
            bureau and normalize_bureau_name(bureau) not in bureaus
        ):
            merged.append(acc.get("name", ""))
    if merged:
        issues.append(("MERGED_ACCOUNTS", merged))

    return issues


def _build_remediation_hint(issue_code: str, items: List[str]) -> str:
    """Generate a short hint text to address ``issue_code``."""

    if issue_code == "MISSING_EXPECTED_ACCOUNT":
        joined = ", ".join(items)
        return (
            "The report segment contains account headings that were missed in your "
            f"analysis: {joined}. Ensure each of these accounts is analyzed and "
            "included separately in the JSON output."
        )
    if issue_code == "MERGED_ACCOUNTS":
        joined = ", ".join(items)
        return (
            f"Some accounts appear merged or list multiple bureaus ({joined}). "
            "List each account separately and only include information for the current bureau."
        )
    return ""


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
    missing_bureaus = [b for b in BUREAUS if b not in segments]

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
    needs_review = bool(missing_bureaus)

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
        while attempt < 3:
            attempt += 1
            headings: List[tuple[str, str]] = []
            attempt_start = time.time()
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
                    data, error_code = analyze_bureau(
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
            attempt_latency_ms = (time.time() - attempt_start) * 1000
            if error_code:
                log_bureau_failure(
                    error_code=error_code,
                    bureau=bureau,
                    expected_headings=len(headings),
                    found_accounts=len(data.get("all_accounts", [])),
                    tokens=0,
                    latency=attempt_latency_ms,
                )

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

        tokens_total_in = 0
        tokens_total_out = 0

        if not error_code:
            issues = _detect_segment_issues(seg_text, data, bureau)
            passes = 0
            while issues and passes < FLAGS.max_remediation_passes:
                issue_code, items = issues[0]
                hint = _build_remediation_hint(issue_code, items)
                rem_prompt = prompt + "\n\n" + hint
                logging.info(
                    "analysis_remediation",
                    extra={"bureau": bureau, "issue": issue_code, "pass": passes + 1},
                )
                new_data, new_err = analyze_bureau(
                    seg_text,
                    is_identity_theft=is_identity_theft,
                    output_json_path=seg_path,
                    ai_client=ai_client,
                    strategic_context=strategic_context,
                    prompt=rem_prompt,
                    late_summary_text=late_summary_text,
                    inquiry_summary=inquiry_summary,
                )
                if new_err:
                    logging.warning(
                        "analysis_remediation_failed",
                        extra={
                            "bureau": bureau,
                            "issue": issue_code,
                            "error_code": new_err,
                        },
                    )
                    break

                for key in [
                    "negative_accounts",
                    "open_accounts_with_issues",
                    "positive_accounts",
                    "high_utilization_accounts",
                    "all_accounts",
                ]:
                    if new_data.get(key):
                        if issue_code == "MERGED_ACCOUNTS":
                            data[key] = new_data[key]
                        else:
                            _merge_accounts(data.setdefault(key, []), new_data[key])

                if new_data.get("inquiries"):
                    _merge_inquiries(
                        data.setdefault("inquiries", []), new_data["inquiries"]
                    )

                for key in [
                    "personal_info_issues",
                    "account_inquiry_matches",
                    "strategic_recommendations",
                ]:
                    if new_data.get(key):
                        data.setdefault(key, []).extend(new_data[key])

                issues = _detect_segment_issues(seg_text, data, bureau)
                passes += 1

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
                "tokens_in": tokens_total_in,
                "tokens_out": tokens_total_out,
                "latency_ms": latency_ms,
                "error_code": error_code or 0,
                "attempts": attempt,
            },
        )
        cost = (
            tokens_total_in * _INPUT_COST_PER_TOKEN
            + tokens_total_out * _OUTPUT_COST_PER_TOKEN
        )
        log_ai_request(tokens_total_in, tokens_total_out, cost, latency_ms)

        if error_code or data.get("confidence", 1) < 0.7:
            needs_review = True

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
    aggregate["needs_human_review"] = needs_review
    if missing_bureaus:
        aggregate["missing_bureaus"] = missing_bureaus
    else:
        aggregate["missing_bureaus"] = []

    return aggregate
