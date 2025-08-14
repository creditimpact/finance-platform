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
    """Evaluate ``normalized_statement`` and ``account_facts`` against rules."""

    emit_counter("stage_2_5.rules_applied")

    # Build accessors for rulebook data
    rules = getattr(rulebook, "rules", None)
    if rules is None and isinstance(rulebook, Mapping):
        rules = rulebook.get("rules", [])

    flags = getattr(rulebook, "flags", None)
    if flags is None and isinstance(rulebook, Mapping):
        flags = rulebook.get("flags", {})

    precedence = getattr(rulebook, "precedence", None)
    if precedence is None and isinstance(rulebook, Mapping):
        precedence = rulebook.get("precedence", [])

    exclusions = getattr(rulebook, "exclusions", None)
    if exclusions is None and isinstance(rulebook, Mapping):
        exclusions = rulebook.get("exclusions", {})

    red_flags: list[str] = []
    if "late" in normalized_statement.lower():
        red_flags.append("late_payment")

    def get_value(path: str) -> Any:
        if path == "statement" or path == "normalized_statement":
            return normalized_statement
        if path.startswith("flags."):
            target = flags
            for part in path.split(".")[1:]:
                if isinstance(target, Mapping):
                    target = target.get(part)
                else:
                    return None
            return target
        target: Any = account_facts
        for part in path.split("."):
            if isinstance(target, Mapping):
                target = target.get(part)
            else:
                return None
        return target

    def eval_cond(cond: Mapping[str, Any]) -> bool:
        if "all" in cond:
            return all(eval_cond(c) for c in cond["all"])
        if "any" in cond:
            return any(eval_cond(c) for c in cond["any"])
        field = cond.get("field", "")
        value = get_value(field)
        if "eq" in cond:
            return value == cond["eq"]
        if "ne" in cond:
            return value != cond["ne"]
        if "lt" in cond:
            try:
                return value < cond["lt"]
            except TypeError:
                return False
        if "lte" in cond:
            try:
                return value <= cond["lte"]
            except TypeError:
                return False
        if "gt" in cond:
            try:
                return value > cond["gt"]
            except TypeError:
                return False
        if "gte" in cond:
            try:
                return value >= cond["gte"]
            except TypeError:
                return False
        return False

    triggered: Dict[str, Mapping[str, Any]] = {}
    for rule in rules or []:
        when = rule.get("when")
        if when and eval_cond(when):
            triggered[rule["id"]] = rule.get("effect", {})

    precedence_map = {rid: i for i, rid in enumerate(precedence or [])}
    sorted_hits = sorted(
        triggered.items(), key=lambda item: precedence_map.get(item[0], len(precedence_map))
    )

    final_hits: list[str] = []
    needs_evidence: list[str] = []
    suggested_dispute_frame = ""
    suppressed: set[str] = set()

    for rule_id, effect in sorted_hits:
        if rule_id in suppressed:
            continue
        final_hits.extend(effect.get("rule_hits", [rule_id]))
        needs_evidence.extend(effect.get("needs_evidence", []))
        if not suggested_dispute_frame and effect.get("suggested_dispute_frame"):
            suggested_dispute_frame = effect["suggested_dispute_frame"]
        for ex in (exclusions or {}).get(rule_id, []):
            suppressed.add(ex)

    # Deduplicate while preserving order
    seen_hits: set[str] = set()
    final_hits = [x for x in final_hits if not (x in seen_hits or seen_hits.add(x))]
    seen_ev: set[str] = set()
    needs_evidence = [x for x in needs_evidence if not (x in seen_ev or seen_ev.add(x))]

    return {
        "rule_hits": final_hits,
        "needs_evidence": needs_evidence,
        "red_flags": red_flags,
        "suggested_dispute_frame": suggested_dispute_frame,
    }


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
