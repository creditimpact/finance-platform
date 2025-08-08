"""Centralized compliance pipeline for rendered documents."""

from __future__ import annotations

import re
from typing import Optional

from logic.guardrails import fix_draft_with_guardrails
from services.ai_client import AIClient

# Re-export existing compliance helpers for compatibility
from .compliance_adapter import (
    adapt_gpt_output,
    sanitize_client_info,
    sanitize_disputes,
    DEFAULT_DISPUTE_REASON,
    ESCALATION_NOTE,
)


def run_compliance_pipeline(
    html: str,
    state: Optional[str],
    session_id: str,
    doc_type: str,
    *,
    ai_client: AIClient | None = None,
) -> str:
    """Apply shared compliance checks to text destined for rendering.

    The pipeline strips HTML tags and routes the plain text through the
    guardrails checker. The original HTML is returned unchanged.
    """

    plain_text = re.sub(r"<[^>]+>", " ", html)
    fix_draft_with_guardrails(
        plain_text,
        state,
        {},
        session_id,
        doc_type,
        ai_client=ai_client,
    )
    return html


# Backwards compatible alias
apply_text_compliance = run_compliance_pipeline

__all__ = [
    "run_compliance_pipeline",
    "apply_text_compliance",
    "adapt_gpt_output",
    "sanitize_client_info",
    "sanitize_disputes",
    "DEFAULT_DISPUTE_REASON",
    "ESCALATION_NOTE",
]
