"""Backward-compatible orchestrator for goodwill letter generation."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

from audit import AuditLogger
from services.ai_client import AIClient
from session_manager import get_session

import logic.goodwill_preparation as goodwill_preparation
import logic.goodwill_prompting as goodwill_prompting
import logic.goodwill_rendering as goodwill_rendering
from logic import pdf_renderer
from logic.utils.pdf_ops import gather_supporting_docs
from logic.compliance_pipeline import run_compliance_pipeline

# ---------------------------------------------------------------------------
# Orchestrator functions
# ---------------------------------------------------------------------------


def generate_goodwill_letter_with_ai(
    creditor: str,
    accounts: list[dict],
    client_info: Dict[str, Any],
    output_path: Path,
    run_date: str | None = None,
    audit: AuditLogger | None = None,
    *,
    ai_client: AIClient | None = None,
) -> None:
    """Generate a single goodwill letter for ``creditor``."""

    session_id = client_info.get("session_id")
    session = get_session(session_id or "") or {}
    structured_summaries = session.get("structured_summaries", {})

    account_summaries = goodwill_preparation.prepare_account_summaries(
        accounts,
        structured_summaries,
        client_info.get("state"),
        audit=audit,
        ai_client=ai_client,
    )

    gpt_data, _ = goodwill_prompting.generate_goodwill_letter_draft(
        client_info.get("legal_name") or client_info.get("name", "Your Name"),
        creditor,
        account_summaries,
        client_info.get("story"),
        client_info.get("tone", "neutral"),
        session_id,
        ai_client=ai_client,
        audit=audit,
    )

    _, doc_names, _ = gather_supporting_docs(session_id or "")

    goodwill_rendering.render_goodwill_letter(
        creditor,
        gpt_data,
        client_info,
        output_path,
        run_date,
        doc_names=doc_names,
        ai_client=ai_client,
        audit=audit,
        compliance_fn=run_compliance_pipeline,
        pdf_fn=pdf_renderer.render_html_to_pdf,
    )


def generate_goodwill_letters(
    client_info: Dict[str, Any],
    bureau_data: Dict[str, Any],
    output_path: Path,
    audit: AuditLogger | None,
    run_date: str | None = None,
    *,
    ai_client: AIClient | None = None,
) -> None:
    """Generate goodwill letters for all eligible creditors in ``bureau_data``."""

    goodwill_accounts = goodwill_preparation.select_goodwill_candidates(
        client_info, bureau_data
    )
    for creditor, accounts in goodwill_accounts.items():
        generate_goodwill_letter_with_ai(
            creditor,
            accounts,
            client_info,
            output_path,
            run_date,
            audit,
            ai_client=ai_client,
        )


__all__ = [
    "generate_goodwill_letter_with_ai",
    "generate_goodwill_letters",
]
