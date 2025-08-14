from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Mapping, Protocol, Tuple

from jsonschema import Draft7Validator, ValidationError

from backend.audit.audit import emit_event
from backend.analytics.analytics_tracker import emit_counter
from backend.core.logic.utils.pii import redact_pii


class Rulebook(Protocol):
    """Protocol representing a rulebook with a version attribute."""

    version: str


# Regex patterns to detect admissions and their corresponding red flags and summaries
ADMISSION_PATTERNS: Tuple[Tuple[re.Pattern[str], str, str], ...] = (
    (
        re.compile(r"\bmy fault\b", re.IGNORECASE),
        "admission_of_fault",
        "Creditor reports an issue; consumer requests verification.",
    ),
    (
        re.compile(r"\bi owe\b", re.IGNORECASE),
        "admission_of_debt",
        "Creditor reports a debt; consumer requests verification.",
    ),
    (
        re.compile(r"\bpagu[eé] tarde\b", re.IGNORECASE),
        "late_payment",
        "Creditor reports a late payment; consumer requests verification.",
    ),
)

_SCHEMA_PATH = Path(__file__).with_name("stage_2_5_schema.json")
_SCHEMA = json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))
_VALIDATOR = Draft7Validator(_SCHEMA)


def _fill_defaults(data: Dict[str, Any]) -> None:
    """Populate ``data`` with default values from the schema."""

    for key, subschema in _SCHEMA.get("properties", {}).items():
        if key not in data and "default" in subschema:
            data[key] = json.loads(json.dumps(subschema["default"]))


def neutralize_admissions(statement: str) -> Tuple[str, list[str], bool]:
    """Return a legally safe version of ``statement``.

    Matches known admission phrases and rewrites the statement into a
    verification-focused summary. Returns the summary, any red flags detected,
    and whether a prohibited admission was present.
    """

    lowered = statement.lower()
    red_flags: list[str] = []
    summary = statement
    prohibited = False
    for pattern, flag, replacement in ADMISSION_PATTERNS:
        if pattern.search(lowered):
            red_flags.append(flag)
            summary = replacement
            prohibited = True

    if prohibited:
        emit_counter("stage_2_5.admission_neutralized_total")
        emit_event(
            "admission_neutralized",
            {"raw_statement": redact_pii(statement)[:100], "summary": summary},
        )

    return summary, red_flags, prohibited


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
    legal_safe_summary, admission_flags, admission_detected = neutralize_admissions(
        user_statement_raw
    )
    evaluation = evaluate_rules(legal_safe_summary, account_facts, rulebook)
    rulebook_version = getattr(rulebook, "version", "")
    if not rulebook_version and isinstance(rulebook, Mapping):
        rulebook_version = str(rulebook.get("version", ""))

    result = evaluation.copy()
    result.update(
        {
            "legal_safe_summary": legal_safe_summary,
            "prohibited_admission_detected": admission_detected,
            "rulebook_version": rulebook_version,
        }
    )
    result["red_flags"] = list(
        dict.fromkeys(result.get("red_flags", []) + admission_flags)
    )

    _fill_defaults(result)
    _VALIDATOR.validate(result)

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
    "ValidationError",
]
