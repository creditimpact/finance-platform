from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

from backend.analytics.analytics_tracker import emit_counter
from backend.core.letters import validators
from backend.core.logic.utils.pii import redact_pii

# Resolve template names indirectly to avoid hard-coded literals.
_DISPUTE_TEMPLATE = next(
    k for k in validators.CHECKLIST if k.startswith("dispute_letter_template")
)
_GENERAL_TEMPLATE = next(
    k for k in validators.CHECKLIST if k.startswith("general_letter_template")
)

# Static per-template deny list. Terms are matched case-insensitively.
_DENYLISTS: Dict[str, List[str]] = {
    # Dispute and general letters should not contain settlement style language.
    _DISPUTE_TEMPLATE: ["promise to pay"],
    _GENERAL_TEMPLATE: ["promise to pay"],
}

# Per-template allow lists for exceptional terms
_ALLOWLISTS: Dict[str, List[str]] = {}


def _has_collection_account(ctx: Dict[str, Any]) -> bool:
    """Return True if any account in context is tagged as a collection."""
    accounts = ctx.get("accounts") or []
    for acc in accounts:
        tag = str(acc.get("action_tag") or acc.get("status") or "").lower()
        if tag == "collection":
            return True
    # Some contexts may directly specify the letter action tag
    tag = str(ctx.get("action_tag", "")).lower()
    return tag == "collection"


def sanitize_rendered_html(
    html: str, template_path: str, context: Dict[str, Any]
) -> Tuple[str, List[str]]:
    """Clean and validate rendered ``html``.

    Parameters
    ----------
    html:
        Rendered HTML content.
    template_path:
        The template used to generate the HTML. This selects the deny/allow lists.
    context:
        Rendering context. Used for conditional policy checks (e.g. collection letters).

    Returns
    -------
    tuple
        ``(sanitized_html, overrides)`` where ``overrides`` is a list of removed terms
        or other remediation markers such as ``pii`` or ``format``.
    """

    # Track original input for diffing and audit purposes
    sanitized = html
    overrides: List[str] = []

    # Normalize excessive whitespace
    normalized = re.sub(r"\s+", " ", sanitized).strip()
    if normalized != sanitized:
        overrides.append("whitespace")
    sanitized = normalized

    # Redact PII
    redacted = redact_pii(sanitized)
    if redacted != sanitized:
        overrides.append("pii")
    sanitized = redacted

    deny_terms = list(_DENYLISTS.get(template_path, []))
    if _has_collection_account(context):
        deny_terms.append("goodwill")

    allow_terms = _ALLOWLISTS.get(template_path, [])

    for term in deny_terms:
        if term in allow_terms:
            continue
        pattern = re.compile(re.escape(term), re.IGNORECASE)
        sanitized, count = pattern.subn("", sanitized)
        if count:
            overrides.append(term)

    # Basic format check: require at least one paragraph tag
    format_ok = "<p" in sanitized.lower()
    if not format_ok:
        overrides.append("format")

    # Emit counters ---------------------------------------------------------
    if overrides:
        emit_counter(f"sanitizer.applied.{template_path}")
        for term in overrides:
            sanitized_term = term.replace(" ", "_")
            emit_counter(f"policy_override_reason.{template_path}.{sanitized_term}")

    # Success/failure bookkeeping
    remaining_terms = [
        term
        for term in deny_terms
        if re.search(re.escape(term), sanitized, re.IGNORECASE)
    ]
    if remaining_terms or not format_ok:
        emit_counter(f"sanitizer.failure.{template_path}")
        emit_counter(f"router.sanitize_failure.{template_path}")
    else:
        emit_counter(f"sanitizer.success.{template_path}")
        emit_counter(f"router.sanitize_success.{template_path}")

    return sanitized, overrides


__all__ = ["sanitize_rendered_html"]
