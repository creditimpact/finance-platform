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
from typing import Any, List, Mapping

from backend.audit.audit import AuditLevel, AuditLogger
from backend.core.logic.utils.note_handling import get_client_address_lines

from backend.core.logic.utils.names_normalization import normalize_creditor_name
from backend.core.logic.strategy.strategy_engine import generate_strategy
from backend.core.logic.guardrails.summary_validator import (
    validate_structured_summaries,
)
from .dispute_preparation import prepare_disputes_and_inquiries
from .gpt_prompting import call_gpt_dispute_letter as _call_gpt_dispute_letter
from backend.core.logic.strategy.summary_classifier import ClassificationRecord
from backend.core.models.letter import LetterContext, LetterArtifact, LetterAccount
from backend.core.models.account import Account, Inquiry
from backend.core.models import ClientInfo, BureauPayload
from backend.core.services.ai_client import AIClient
from backend.core.logic.rendering.letter_rendering import render_dispute_letter_html
from backend.core.logic.rendering import pdf_renderer
from backend.core.logic.compliance.compliance_pipeline import (
    run_compliance_pipeline,
    adapt_gpt_output,
    sanitize_client_info,
    sanitize_disputes,
    DEFAULT_DISPUTE_REASON,
    ESCALATION_NOTE,
)


logger = logging.getLogger(__name__)


CREDIT_BUREAU_ADDRESSES = {
    "Experian": "P.O. Box 4500, Allen, TX 75013",
    "Equifax": "P.O. Box 740256, Atlanta, GA 30374-0256",
    "TransUnion": "P.O. Box 2000, Chester, PA 19016-2000",
}


def call_gpt_dispute_letter(
    *args,
    audit: AuditLogger | None = None,
    classification_map=None,
    **kwargs,
):
    """Proxy to GPT prompting module for backward compatibility."""
    ctx = _call_gpt_dispute_letter(
        *args, audit=audit, classification_map=classification_map, **kwargs
    )
    return ctx.to_dict()


def generate_all_dispute_letters_with_ai(
    client: ClientInfo,
    bureau_map: Mapping[str, BureauPayload | dict[str, Any]],
    output_path: Path,
    is_identity_theft: bool,
    audit: AuditLogger | None,
    ai_client: AIClient,
    *,
    classification_map: Mapping[str, ClassificationRecord] | None = None,
    run_date: str | None = None,
    log_messages: List[str] | None = None,
    rulebook_fallback_enabled: bool = True,
    wkhtmltopdf_path: str | None = None,
):
    """Generate dispute letters for all bureaus using GPT-derived content."""

    if isinstance(client, dict):  # pragma: no cover - backward compat
        client = ClientInfo.from_dict(client)
    client_info = client.to_dict()
    bureau_data = {
        k: (
            (
                BureauPayload.from_dict(v).to_dict()
                if isinstance(v, dict)
                else v.to_dict()
            )
            if isinstance(v, (BureauPayload, dict))
            else v
        )
        for k, v in bureau_map.items()
    }

    output_path.mkdir(parents=True, exist_ok=True)
    if log_messages is None:
        log_messages = []

    account_inquiry_matches = client_info.get("account_inquiry_matches", [])
    client_name = client_info.get("legal_name") or client_info.get("name", "Client")

    if not client_info.get("legal_name"):
        print(
            "[WARN] Warning: legal_name not found in client_info. Using fallback name."
        )

    session_id = client_info.get("session_id", "")
    strategy = generate_strategy(session_id, bureau_data)
    strategy_summaries = validate_structured_summaries(
        strategy.get("dispute_items", {})
    )

    sanitization_issues = False

    for bureau_name, payload in bureau_data.items():
        print(f"[INFO] Generating letter for {bureau_name}...")
        bureau_sanitization = False

        disputes, filtered_inquiries, acc_type_map = prepare_disputes_and_inquiries(
            bureau_name,
            payload,
            client_info,
            account_inquiry_matches,
            log_messages,
        )
        disputes = disputes or []
        filtered_inquiries = filtered_inquiries or []

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
            print(f"[WARN] No data to dispute for {bureau_name}, skipping.")
            log_messages.append(msg)
            continue

        bureau_address = CREDIT_BUREAU_ADDRESSES.get(bureau_name, "Unknown")

        dispute_objs = [
            Account.from_dict(d) if isinstance(d, dict) else d for d in disputes
        ]
        inquiry_objs = [
            Inquiry.from_dict(i) if isinstance(i, dict) else i
            for i in filtered_inquiries
        ]

        gpt_data = call_gpt_dispute_letter(
            client_info_for_gpt,
            bureau_name,
            dispute_objs,
            inquiry_objs,
            is_identity_theft,
            strategy_summaries,
            client_info.get("state", ""),
            classification_map=classification_map,
            audit=audit,
            ai_client=ai_client,
        )

        adapt_gpt_output(
            gpt_data, fallback_norm_names, acc_type_map, rulebook_fallback_enabled
        )

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
                    f"[WARN] Inquiry '{inq.get('creditor_name')}' expected in dispute letter but was not included."
                )

        if run_date is None:
            run_date = datetime.now().strftime("%B %d, %Y")

        context = LetterContext(
            client_name=client_name,
            client_address_lines=get_client_address_lines(client_info),
            bureau_name=bureau_name,
            bureau_address=bureau_address,
            date=run_date,
            opening_paragraph=gpt_data.get("opening_paragraph", ""),
            accounts=[
                LetterAccount.from_dict(a) if isinstance(a, dict) else a
                for a in gpt_data.get("accounts", [])
            ],
            inquiries=[
                Inquiry.from_dict(i) if isinstance(i, dict) else i
                for i in gpt_data.get("inquiries", [])
            ],
            closing_paragraph=gpt_data.get("closing_paragraph", ""),
            is_identity_theft=is_identity_theft,
        )

        artifact = render_dispute_letter_html(context)
        html = artifact.html if isinstance(artifact, LetterArtifact) else artifact
        run_compliance_pipeline(
            artifact,
            client_info.get("state"),
            client_info.get("session_id", ""),
            "dispute",
            ai_client=ai_client,
        )
        filename = f"Dispute Letter - {bureau_name}.pdf"
        filepath = output_path / filename
        if wkhtmltopdf_path:
            pdf_renderer.render_html_to_pdf(
                html, str(filepath), wkhtmltopdf_path=wkhtmltopdf_path
            )
        else:
            pdf_renderer.render_html_to_pdf(html, str(filepath))

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
    "call_gpt_dispute_letter",
]
