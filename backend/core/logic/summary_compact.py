from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any


_MERGE_SCORING_ALLOWED = {
    "best_with",
    "score_total",
    "reasons",
    "conflicts",
    "identity_score",
    "debt_score",
    "acctnum_level",
    "matched_fields",
    "acctnum_digits_len_a",
    "acctnum_digits_len_b",
}

_MERGE_EXPLANATION_ALLOWED = {
    "kind",
    "with",
    "decision",
    "total",
    "parts",
    "matched_fields",
    "reasons",
    "conflicts",
    "strong",
    "acctnum_level",
    "acctnum_digits_len_a",
    "acctnum_digits_len_b",
}

_BANNED_KEYS = {
    "aux",
    "by_field_pairs",
    "matched_pairs",
    "tiebreaker",
    "strong_rank",
    "dates_all",
    "mid",
}


def _ensure_bool_mapping(value: Any) -> Any:
    """Return a mapping containing only boolean values when possible."""

    if isinstance(value, Mapping):
        return {key: bool(val) for key, val in value.items()}
    return value


def _filter_keys(source: Mapping[str, Any], allowed_keys: set[str]) -> dict[str, Any]:
    """Return a shallow copy containing only allowed keys."""

    filtered: dict[str, Any] = {}
    for key, value in source.items():
        if key not in allowed_keys:
            continue
        if key == "matched_fields":
            value = _ensure_bool_mapping(value)

        filtered[key] = deepcopy(value)

    return filtered


def _scrub_banned(value: Any) -> Any:
    """Recursively remove banned keys from dictionaries."""

    if isinstance(value, Mapping):
        cleaned: dict[str, Any] = {}
        for key, item in value.items():
            if key in _BANNED_KEYS:
                continue
            cleaned[key] = _scrub_banned(item)
        return cleaned

    if isinstance(value, list):
        return [_scrub_banned(item) for item in value]

    if isinstance(value, tuple):
        return tuple(_scrub_banned(item) for item in value)

    return value


def compact_merge_sections(summary: dict[str, Any]) -> dict[str, Any]:
    """Compact merge sections and scrub banned keys from a summary payload."""

    merge_scoring = summary.get("merge_scoring")
    if isinstance(merge_scoring, Mapping):
        summary["merge_scoring"] = _filter_keys(merge_scoring, _MERGE_SCORING_ALLOWED)

    merge_explanations = summary.get("merge_explanations")
    if isinstance(merge_explanations, Sequence) and not isinstance(
        merge_explanations, (str, bytes, bytearray)
    ):
        filtered_explanations = []
        for entry in merge_explanations:
            if not isinstance(entry, Mapping):
                continue
            filtered = _filter_keys(entry, _MERGE_EXPLANATION_ALLOWED)
            filtered_explanations.append(filtered)
        summary["merge_explanations"] = filtered_explanations

    scrubbed = _scrub_banned(summary)
    summary.clear()
    summary.update(scrubbed)
    return summary
