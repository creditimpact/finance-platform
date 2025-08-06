from __future__ import annotations

import re
from typing import TypedDict, Literal

from logic.rules_loader import load_rules, load_state_rules


class RuleViolation(TypedDict):
    rule_id: str
    severity: Literal["critical", "warning"]
    span: tuple[int, int] | None
    message: str


def check_letter(text: str, state: str | None, context: dict) -> tuple[str, list[RuleViolation]]:
    """
    Returns (possibly_fixed_text, violations)
    - Load systemic rules
    - Scan for block_patterns; for each match:
        * If fix_template exists → replace
        * Else → record violation
    - Mask PII according to RULE_PII_LIMIT
    - Append state-specific clauses if applicable
    - Return modified text + list of violations
    """
    rules = load_rules()
    state_rules = load_state_rules()

    modified_text = text
    violations: list[RuleViolation] = []

    # PII masking first
    for rule in rules:
        if rule.get("id") != "RULE_PII_LIMIT":
            continue
        for pattern in rule.get("block_patterns", []):
            regex = re.compile(pattern)
            matches = list(regex.finditer(modified_text))
            for m in matches:
                violations.append(
                    {
                        "rule_id": rule["id"],
                        "severity": rule.get("severity", "warning"),
                        "span": m.span(),
                        "message": rule.get("description", ""),
                    }
                )
            modified_text = regex.sub("[REDACTED]", modified_text)

    # Apply other systemic rules
    for rule in rules:
        if rule.get("id") == "RULE_PII_LIMIT":
            continue
        for pattern in rule.get("block_patterns", []):
            regex = re.compile(pattern, flags=re.IGNORECASE)
            matches = list(regex.finditer(modified_text))
            for m in matches:
                violations.append(
                    {
                        "rule_id": rule["id"],
                        "severity": rule.get("severity", "warning"),
                        "span": m.span(),
                        "message": rule.get("description", ""),
                    }
                )
            if matches and rule.get("fix_template"):
                modified_text = regex.sub(rule["fix_template"], modified_text)

    # Append state-specific clauses
    if state:
        state_data = state_rules.get(state.upper())
        clauses: list[str] = []
        if state_data:
            disclosures = state_data.get("disclosures")
            if disclosures:
                clauses.extend(disclosures)
            if "medical_debt_clause" in state_data and context.get("debt_type") == "medical":
                clauses.append(state_data["medical_debt_clause"])
        if clauses:
            if not modified_text.endswith("\n"):
                modified_text += "\n"
            modified_text += "\n" + "\n".join(clauses)

    return modified_text, violations
