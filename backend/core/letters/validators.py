from __future__ import annotations

from typing import Dict, List
import re
from backend.core.logic.utils.pii import redact_pii


CHECKLIST: Dict[str, List[str]] = {
    "dispute_letter_template.html": ["bureau"],
    "goodwill_letter_template.html": ["creditor"],
    "general_letter_template.html": ["recipient"],
    "instruction_template.html": [
        "client_name",
        "date",
        "accounts_summary",
        "per_account_actions",
    ],
}


def validate_required_fields(
    template_path: str | None,
    ctx: dict,
    required: List[str],
    checklist: Dict[str, List[str]],
) -> List[str]:
    """Return missing required fields for ``template_path``."""

    expected = required or checklist.get(template_path or "", [])
    missing = [field for field in expected if not ctx.get(field)]

    sentence = ctx.get("client_context_sentence")
    if sentence:
        if len(sentence) > 150:
            missing.append("client_context_sentence.length")
        if sentence != redact_pii(sentence):
            missing.append("client_context_sentence.pii")
        if re.search(r"promise to pay", sentence, re.IGNORECASE):
            missing.append("client_context_sentence.banned")

    if template_path == "instruction_template.html":
        actions = ctx.get("per_account_actions") or []
        if not actions:
            missing.append("per_account_actions")
        else:
            for action in actions:
                if not action.get("account_ref"):
                    missing.append("per_account_actions.account_ref")
                    break
                sentence = action.get("action_sentence", "")
                if not sentence:
                    missing.append("per_account_actions.action_sentence")
                    break
                if not re.search(
                    r"\b(pay|send|contact|review|dispute|call|update|keep|monitor|mail)\b",
                    sentence,
                    re.IGNORECASE,
                ):
                    missing.append("per_account_actions.action_verb")
                    break


    return missing


__all__ = ["validate_required_fields", "CHECKLIST"]

