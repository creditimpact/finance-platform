from __future__ import annotations

from typing import Any, Dict, Mapping, Protocol

from backend.audit.audit import emit_event


class Rulebook(Protocol):
    """Protocol representing a rulebook with a version attribute."""

    version: str


def neutralize_admissions(statement: str) -> str:
    """Return a legally safe version of ``statement``.

    This is currently a stub that returns the statement unchanged.
    """

    return statement


def evaluate_rules(
    normalized_statement: str, account_facts: Dict[str, Any], rulebook: Rulebook
) -> Dict[str, list]:
    """Evaluate ``normalized_statement`` against the ``rulebook``.

    This stub returns empty results.
    """

    return {"rule_hits": [], "needs_evidence": [], "red_flags": []}


def normalize_and_tag(
    account_cls: Dict[str, Any],
    account_facts: Dict[str, Any],
    rulebook: Rulebook,
    account_id: str | None = None,
) -> Dict[str, Any]:
    """Normalize user statements and tag accounts with rulebook metadata."""

    user_statement_raw = (
        account_cls.get("user_statement_raw")
        or account_facts.get("user_statement_raw")
        or "No statement provided"
    )
    legal_safe_summary = neutralize_admissions(user_statement_raw)
    evaluation = evaluate_rules(legal_safe_summary, account_facts, rulebook)
    rulebook_version = getattr(rulebook, "version", "")
    if not rulebook_version and isinstance(rulebook, Mapping):
        rulebook_version = str(rulebook.get("version", ""))

    result = {
        "legal_safe_summary": legal_safe_summary,
        "suggested_dispute_frame": "",
        "rule_hits": evaluation.get("rule_hits", []),
        "needs_evidence": evaluation.get("needs_evidence", []),
        "red_flags": evaluation.get("red_flags", []),
        "rulebook_version": rulebook_version,
    }
    if account_id:
        emit_event(
            "rule_evaluated",
            {
                "account_id": account_id,
                "rule_hits": result["rule_hits"],
                "rulebook_version": rulebook_version,
            },
        )
    return result


__all__ = [
    "normalize_and_tag",
    "neutralize_admissions",
    "evaluate_rules",
    "Rulebook",
]
