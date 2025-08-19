from __future__ import annotations

from typing import Dict, List


CHECKLIST: Dict[str, List[str]] = {
    "dispute_letter_template.html": ["bureau"],
    "goodwill_letter_template.html": ["creditor"],
    "general_letter_template.html": ["recipient"],
}


def validate_required_fields(
    template_path: str | None,
    ctx: dict,
    required: List[str],
    checklist: Dict[str, List[str]],
) -> List[str]:
    """Return missing required fields for ``template_path``."""

    expected = required or checklist.get(template_path or "", [])
    return [field for field in expected if not ctx.get(field)]


__all__ = ["validate_required_fields", "CHECKLIST"]

