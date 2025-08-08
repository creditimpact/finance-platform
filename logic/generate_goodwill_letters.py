"""Backward-compatible orchestrator for goodwill letter generation."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any
import warnings

from audit import AuditLogger
from services.ai_client import AIClient
from session_manager import get_session

from . import goodwill_preparation, goodwill_prompting, goodwill_rendering
from logic.utils.names_normalization import normalize_creditor_name as _normalize_creditor_name
from logic.pdf_renderer import render_html_to_pdf as _render_html_to_pdf
from logic.utils.pdf_ops import gather_supporting_docs
from logic.compliance_pipeline import run_compliance_pipeline as _run_compliance_pipeline

# ---------------------------------------------------------------------------
# Deprecated helpers
# ---------------------------------------------------------------------------

def normalize_creditor_name(raw_name: str) -> str:  # pragma: no cover - thin wrapper
    """Deprecated wrapper for ``utils.names_normalization.normalize_creditor_name``."""
    warnings.warn(
        "normalize_creditor_name has moved to logic.utils.names_normalization; "
        "import from there instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _normalize_creditor_name(raw_name)


def call_gpt_for_goodwill_letter(*args, **kwargs):  # pragma: no cover - wrapper
    """Deprecated wrapper forwarding to :func:`goodwill_prompting.generate_goodwill_letter_draft`."""
    warnings.warn(
        "call_gpt_for_goodwill_letter has moved to logic.goodwill_prompting; "
        "import from there instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    result, _ = goodwill_prompting.generate_goodwill_letter_draft(*args, **kwargs)
    return result


def render_html_to_pdf(html: str, output_path: Path):  # pragma: no cover - wrapper
    """Deprecated wrapper forwarding to :mod:`logic.pdf_renderer`."""
    warnings.warn(
        "render_html_to_pdf has moved to logic.pdf_renderer; import from there instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _render_html_to_pdf(html, str(output_path))


def load_creditor_address_map():  # pragma: no cover - wrapper
    """Deprecated wrapper forwarding to :mod:`logic.goodwill_rendering`."""
    warnings.warn(
        "load_creditor_address_map has moved to logic.goodwill_rendering; import from there instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return goodwill_rendering.load_creditor_address_map()


def run_compliance_pipeline(html, state, session_id, doc_type, *, ai_client=None):  # pragma: no cover - wrapper
    """Expose compliance pipeline for backwards compatibility."""
    return _run_compliance_pipeline(html, state, session_id, doc_type, ai_client=ai_client)

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

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        gpt_data = call_gpt_for_goodwill_letter(
            client_info.get("legal_name") or client_info.get("name", "Your Name"),
            creditor,
            account_summaries,
            client_info.get("story"),
            client_info.get("tone", "neutral"),
            session_id,
            structured_summaries,
            client_info.get("state"),
            audit,
            ai_client,
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
        pdf_fn=render_html_to_pdf,
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
    "normalize_creditor_name",
    "call_gpt_for_goodwill_letter",
    "render_html_to_pdf",
    "load_creditor_address_map",
    "run_compliance_pipeline",
]
