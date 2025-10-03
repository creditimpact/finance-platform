"""Utilities for classifying bureau disagreement scenarios.

This module provides helpers that summarize the shape of a field's bureau
responses.  The primary entry point is :func:`classify_reason`, which accepts a
mapping of bureau identifiers to their values (raw or normalized) and
determines the appropriate escalation reason code.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping

__all__ = ["classify_reason"]


_REASON_LABELS: Dict[str, str] = {
    "C1_TWO_PRESENT_ONE_MISSING": "two present, one missing",
    "C2_ONE_MISSING": "only one bureau reported a value",
    "C3_TWO_PRESENT_CONFLICT": "conflict with one bureau missing",
    "C4_TWO_MATCH_ONE_DIFF": "two bureaus agree, one differs",
    "C5_ALL_DIFF": "all bureaus reported different values",
    "C6_ALL_MISSING": "all bureaus missing value",
}


def _is_missing(value: Any) -> bool:
    """Return ``True`` when ``value`` should be treated as missing."""

    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() in {"", "--"}
    return False


def _freeze(value: Any) -> Any:
    """Convert ``value`` into a hashable representation for comparisons."""

    if isinstance(value, dict):
        return tuple(sorted((str(k), _freeze(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple, set, frozenset)):
        return tuple(_freeze(item) for item in value)
    return value


def classify_reason(bureau_values: Mapping[str, Any]) -> Mapping[str, Any]:
    """Classify the disagreement pattern for ``bureau_values``.

    Parameters
    ----------
    bureau_values:
        Mapping of bureau identifiers (e.g., ``"experian"``) to the values they
        reported for a field. Values may be raw or normalized representations.

    Returns
    -------
    Mapping[str, Any]
        A dictionary containing the reason code, metadata about missing vs
        mismatch counts, and helper booleans that downstream callers can use.
    """

    total_bureaus = len(bureau_values)
    missing_count = 0
    present_values: list[Any] = []

    for value in bureau_values.values():
        if _is_missing(value):
            missing_count += 1
        else:
            present_values.append(value)

    present_count = total_bureaus - missing_count
    distinct_values = len({_freeze(value) for value in present_values})

    is_missing = missing_count > 0
    is_mismatch = distinct_values > 1

    if present_count == 0:
        reason_code = "C6_ALL_MISSING"
    elif present_count == 1:
        reason_code = "C2_ONE_MISSING"
    elif missing_count > 0:
        if distinct_values <= 1:
            reason_code = "C1_TWO_PRESENT_ONE_MISSING"
        else:
            reason_code = "C3_TWO_PRESENT_CONFLICT"
    else:
        if distinct_values <= 1:
            # This scenario should already be filtered out before classification,
            # but fall back to C4 for completeness.
            reason_code = "C4_TWO_MATCH_ONE_DIFF"
        elif distinct_values == 2:
            reason_code = "C4_TWO_MATCH_ONE_DIFF"
        else:
            reason_code = "C5_ALL_DIFF"

    return {
        "reason_code": reason_code,
        "reason_label": _REASON_LABELS[reason_code],
        "is_missing": is_missing,
        "is_mismatch": is_mismatch,
        "missing_count": missing_count,
        "present_count": present_count,
        "distinct_values": distinct_values,
    }

