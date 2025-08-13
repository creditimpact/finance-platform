"""Backward-compatible orchestrator for goodwill letter generation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import backend.core.logic.letters.goodwill_preparation as goodwill_preparation
import backend.core.logic.letters.goodwill_prompting as goodwill_prompting
import backend.core.logic.letters.goodwill_rendering as goodwill_rendering
from backend.api.session_manager import get_session
from backend.audit.audit import AuditLogger
from backend.core.logic.compliance.compliance_pipeline import \
    run_compliance_pipeline
from backend.core.logic.guardrails.summary_validator import \
    validate_structured_summaries
from backend.core.logic.rendering import pdf_renderer
from backend.core.logic.utils.pdf_ops import gather_supporting_docs
from backend.core.models import BureauPayload, ClientInfo
from backend.core.models.account import Account
from backend.core.services.ai_client import AIClient

# ---------------------------------------------------------------------------
# Orchestrator functions
# ---------------------------------------------------------------------------


def generate_goodwill_letter_with_ai(
    creditor: str,
    accounts: list[Account],
    client: ClientInfo,
    output_path: Path,
    run_date: str | None = None,
    audit: AuditLogger | None = None,
    *,
    ai_client: AIClient,
) -> None:
    """Generate a single goodwill letter for ``creditor``."""

    if isinstance(client, dict):  # pragma: no cover - backward compat
        client = ClientInfo.from_dict(client)
    account_objs = [
        Account.from_dict(a) if isinstance(a, dict) else a for a in accounts
    ]
    client_info = client.to_dict()
    account_dicts = [a.to_dict() for a in account_objs]
    session_id = client_info.get("session_id")
    session = get_session(session_id or "") or {}
    structured_summaries = session.get("structured_summaries", {})
    structured_summaries = validate_structured_summaries(structured_summaries)

    account_summaries = goodwill_preparation.prepare_account_summaries(
        account_dicts,
        structured_summaries,
        client_info.get("state"),
        session_id,
        audit=audit,
        ai_client=ai_client,
    )

    gpt_data, _ = goodwill_prompting.generate_goodwill_letter_draft(
        client_info.get("legal_name") or client_info.get("name", "Your Name"),
        creditor,
        account_summaries,
        tone=client_info.get("tone", "neutral"),
        session_id=session_id,
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
    client: ClientInfo,
    bureau_map: Mapping[str, BureauPayload | dict[str, Any]],
    output_path: Path,
    audit: AuditLogger | None,
    run_date: str | None = None,
    *,
    ai_client: AIClient,
    identity_theft: bool = False,
) -> None:
    """Generate goodwill letters for all eligible creditors in ``bureau_data``.

    Parameters
    ----------
    identity_theft:
        When ``True`` the function returns immediately without generating any
        letters.  This mirrors the higher level orchestration logic and acts as
        a defensive guard should callers invoke this helper directly.
    """

    if identity_theft:
        return

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

    goodwill_accounts = goodwill_preparation.select_goodwill_candidates(
        client_info, bureau_data
    )
    for creditor, accounts in goodwill_accounts.items():
        account_objs = [
            Account.from_dict(a) if isinstance(a, dict) else a for a in accounts
        ]
        generate_goodwill_letter_with_ai(
            creditor,
            account_objs,
            client,
            output_path,
            run_date,
            audit,
            ai_client=ai_client,
        )


__all__ = [
    "generate_goodwill_letter_with_ai",
    "generate_goodwill_letters",
]
