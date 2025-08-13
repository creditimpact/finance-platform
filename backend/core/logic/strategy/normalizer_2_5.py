from __future__ import annotations

from typing import Any, Dict, Mapping, Protocol

from backend.audit.audit import emit_event
from backend.analytics.analytics_tracker import emit_counter


class Rulebook(Protocol):
    """Protocol representing a rulebook with a version attribute."""

    version: str


def neutralize_admissions(statement: str) -> str:
    """Return a legally safe version of ``statement``.

    Currently performs a minimal replacement of first-person admissions and
    emits a metrics counter when a change occurs.
    """

    lowered = statement.lower()
    if "i was late" in lowered:
        emit_counter("stage_2_5.admission_neutralized")
        return statement.replace("I was", "The consumer was").replace(
            "i was", "The consumer was"
        )
    return statement


def evaluate_rules(
    normalized_statement: str, account_facts: Dict[str, Any], rulebook: Rulebook
) -> Dict[str, list]:
    """Evaluate ``normalized_statement`` against the ``rulebook``.

    Emits a metrics counter and returns basic ``red_flags`` if admissions are
    detected. The rulebook argument is accepted for future expansion.
    """

    emit_counter("stage_2_5.rules_applied")
    red_flags: list[str] = []
    if "late" in normalized_statement.lower():
        red_flags.append("late_payment")
    return {"rule_hits": [], "needs_evidence": [], "red_flags": red_flags}


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
