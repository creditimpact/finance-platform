"""Backward-compatible wrapper for generating dispute letters.

This module now delegates to specialized modules for preparation, GPT prompting,
compliance adjustments, and rendering.
"""

from __future__ import annotations

import json
import logging
import warnings
from datetime import datetime
from pathlib import Path
from typing import List

import config
from audit import AuditLevel, AuditLogger
from logic.utils.note_handling import get_client_address_lines

from logic.utils.names_normalization import normalize_creditor_name
from .strategy_engine import generate_strategy
from .dispute_preparation import prepare_disputes_and_inquiries
from .gpt_prompting import call_gpt_dispute_letter as _call_gpt_dispute_letter
from services.ai_client import AIClient, get_default_ai_client
from .letter_rendering import render_dispute_letter_html, render_html_to_pdf
from .compliance_pipeline import (
    run_compliance_pipeline,
    adapt_gpt_output,
    sanitize_client_info,
    sanitize_disputes,
    DEFAULT_DISPUTE_REASON,
    ESCALATION_NOTE,
)
from .summary_classifier import classify_client_summary  # backward compatibility


logger = logging.getLogger(__name__)


CREDIT_BUREAU_ADDRESSES = {
    "Experian": "P.O. Box 4500, Allen, TX 75013",
    "Equifax": "P.O. Box 740256, Atlanta, GA 30374-0256",
    "TransUnion": "P.O. Box 2000, Chester, PA 19016-2000",
}


def call_gpt_dispute_letter(*args, audit: AuditLogger | None = None, **kwargs):
    """Proxy to GPT prompting module for backward compatibility."""

    return _call_gpt_dispute_letter(
        *args, audit=audit, **kwargs, classifier=classify_client_summary
    )


def generate_all_dispute_letters_with_ai(
    client_info: dict,
    bureau_data: dict,
    output_path: Path,
    is_identity_theft: bool,
    audit: AuditLogger | None,
    run_date: str | None = None,
    log_messages: List[str] | None = None,
    ai_client: AIClient | None = None,
):
    """Generate dispute letters for all bureaus using GPT-derived content."""

    ai_client = ai_client or get_default_ai_client()
    output_path.mkdir(parents=True, exist_ok=True)
    if log_messages is None:
        log_messages = []

    account_inquiry_matches = client_info.get("account_inquiry_matches", [])
    client_name = client_info.get("legal_name") or client_info.get("name", "Client")

    if not client_info.get("legal_name"):
        print("[⚠️] Warning: legal_name not found in client_info. Using fallback name.")

    session_id = client_info.get("session_id", "")
    strategy = generate_strategy(session_id, bureau_data)
    strategy_summaries = strategy.get("dispute_items", {})

    sanitization_issues = False

    for bureau_name, payload in bureau_data.items():
        print(f"[🤖] Generating letter for {bureau_name}...")
        bureau_sanitization = False

        disputes, filtered_inquiries, acc_type_map = prepare_disputes_and_inquiries(
            bureau_name,
            payload,
            client_info,
            account_inquiry_matches,
            log_messages,
        )

        sanitized, bureau_flag, fallback_norm_names, fallback_used = sanitize_disputes(
            disputes,
            bureau_name,
            strategy_summaries,
            log_messages,
            is_identity_theft,
        )
        sanitization_issues |= sanitized
        bureau_sanitization |= bureau_flag

        client_info_for_gpt, raw_client_text_present = sanitize_client_info(
            client_info,
            bureau_name,
            log_messages,
        )
        sanitization_issues |= raw_client_text_present
        bureau_sanitization |= raw_client_text_present

        if not disputes and not filtered_inquiries:
            msg = f"[{bureau_name}] No disputes or inquiries after filtering - letter skipped"
            print(f"[⚠️] No data to dispute for {bureau_name}, skipping.")
            log_messages.append(msg)
            continue

        bureau_address = CREDIT_BUREAU_ADDRESSES.get(bureau_name, "Unknown")

        gpt_data = call_gpt_dispute_letter(
            client_info_for_gpt,
            bureau_name,
            disputes,
            filtered_inquiries,
            is_identity_theft,
            strategy_summaries,
            client_info.get("state", ""),
            audit=audit,
            ai_client=ai_client,
        )

        adapt_gpt_output(gpt_data, fallback_norm_names, acc_type_map)

        included_set = {
            (
                normalize_creditor_name(i.get("creditor_name", "")),
                i.get("date"),
            )
            for i in gpt_data.get("inquiries", [])
        }
        for inq in filtered_inquiries:
            key = (
                normalize_creditor_name(inq.get("creditor_name", "")),
                inq.get("date"),
            )
            if key not in included_set:
                print(
                    f"[⚠️] Inquiry '{inq.get('creditor_name')}' expected in dispute letter but was not included."
                )

        if run_date is None:
            run_date = datetime.now().strftime("%B %d, %Y")

        context = {
            "client_name": client_name,
            "client_street": client_info.get("street", ""),
            "client_city": client_info.get("city", ""),
            "client_state": client_info.get("state", ""),
            "client_zip": client_info.get("zip", ""),
            "client_address_lines": get_client_address_lines(client_info),
            "bureau_name": bureau_name,
            "bureau_address": bureau_address,
            "date": run_date,
            "opening_paragraph": gpt_data.get("opening_paragraph", ""),
            "accounts": gpt_data.get("accounts", []),
            "inquiries": gpt_data.get("inquiries", []),
            "closing_paragraph": gpt_data.get("closing_paragraph", ""),
            "is_identity_theft": is_identity_theft,
        }

        html = render_dispute_letter_html(context)
        run_compliance_pipeline(
            html,
            client_info.get("state"),
            client_info.get("session_id", ""),
            "dispute",
            ai_client=ai_client,
        )
        filename = f"Dispute Letter - {bureau_name}.pdf"
        filepath = output_path / filename
        render_html_to_pdf(html, filepath)

        with open(output_path / f"{bureau_name}_gpt_response.json", "w") as f:
            json.dump(gpt_data, f, indent=2)

        if audit and audit.level is AuditLevel.VERBOSE:
            audit.log_step(
                "dispute_letter_generated",
                {
                    "bureau": bureau_name,
                    "output_pdf": str(filepath),
                    "response": gpt_data,
                    "fallback_used": fallback_used,
                    "raw_client_text_present": raw_client_text_present,
                    "sanitization_issues": bureau_sanitization,
                },
            )

        if bureau_sanitization or fallback_used or raw_client_text_present:
            warnings.warn(
                f"[Alert] Issues detected generating letter for {bureau_name}",
                stacklevel=2,
            )

        print(
            f"[LetterSummary] letter_id={filename}, bureau={bureau_name}, action=dispute, "
            f"template=dispute_letter_template.html, fallback_used={fallback_used}, "
            f"raw_client_text_present={raw_client_text_present}, sanitization_issues={bureau_sanitization}"
        )


generate_dispute_letters_for_all_bureaus = generate_all_dispute_letters_with_ai


__all__ = [
    "generate_all_dispute_letters_with_ai",
    "generate_dispute_letters_for_all_bureaus",
    "DEFAULT_DISPUTE_REASON",
    "ESCALATION_NOTE",
    "classify_client_summary",
    "call_gpt_dispute_letter",
]

