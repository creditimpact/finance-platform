"""Reporting comparison utilities for deterministic escalation."""
from __future__ import annotations

from typing import Any, Dict

from .eligibility_policy import canonicalize_history


def classify_reporting_pattern(values_by_bureau: Dict[str, Any]) -> str:
    """Classify the reporting pattern among bureau values.

    Args:
        values_by_bureau: Mapping of bureau name to the raw reported value.

    Returns:
        One of ``case_1`` through ``case_6`` describing the combination of
        missing and present values along with their equality relationships.
    """

    canonicalized: Dict[str, str | None] = {
        bureau: canonicalize_history(value)
        for bureau, value in values_by_bureau.items()
    }

    present_values = [value for value in canonicalized.values() if value is not None]
    missing_count = len(values_by_bureau) - len(present_values)

    if not present_values:
        return "case_6"

    if len(present_values) == 1:
        return "case_1"

    if missing_count == 1:
        first, second = present_values
        if first == second:
            return "case_2"
        return "case_3"

    # At this point all bureaus have a value (missing_count == 0).
    unique_values = set(present_values)
    if len(unique_values) <= 2:
        return "case_4"
    return "case_5"
