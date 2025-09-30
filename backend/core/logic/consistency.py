"""Utilities for cross-bureau field consistency detection."""

from __future__ import annotations

import re
from typing import Any, Dict, Mapping, MutableMapping

__all__ = ["compute_inconsistent_fields"]


_BUREAU_KEYS = ("transunion", "experian", "equifax")
_MONEY_FIELDS = {
    "balance_owed",
    "past_due_amount",
    "high_balance",
    "credit_limit",
    "payment_amount",
}
_DATE_FIELDS = {
    "date_opened",
    "closed_date",
    "last_payment",
    "date_of_last_activity",
    "date_reported",
    "last_verified",
}
_MISSING_SENTINELS = {None, "", "--"}

_AMOUNT_SANITIZE_RE = re.compile(r"[,$\s]")
_AMOUNT_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _is_missing(value: Any) -> bool:
    if value in _MISSING_SENTINELS:
        return True
    if isinstance(value, str) and not value.strip():
        return True
    return False


def _normalize_money(value: Any) -> float | None:
    if _is_missing(value):
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)

    text = str(value).strip()
    if not text:
        return None

    negative = False
    if text.startswith("(") and text.endswith(")"):
        negative = True
        text = text[1:-1]

    cleaned = _AMOUNT_SANITIZE_RE.sub("", text)
    match = _AMOUNT_RE.search(cleaned)
    if not match:
        return None

    try:
        number = float(match.group())
    except ValueError:
        return None

    if negative and number >= 0:
        number = -number
    return number


def _normalize_text(value: Any) -> str | None:
    if _is_missing(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    normalized = " ".join(text.split()).lower()
    return normalized or None


def _normalize_date(value: Any) -> str | None:
    return _normalize_text(value)


def _normalize_value(field: str, value: Any) -> Any:
    if field in _MONEY_FIELDS:
        return _normalize_money(value)
    if field in _DATE_FIELDS or field.endswith("_date"):
        return _normalize_date(value)
    return _normalize_text(value)


def compute_inconsistent_fields(bureaus: Mapping[str, Mapping[str, Any]]) -> Dict[str, Dict[str, MutableMapping[str, Any]]]:
    """Return fields whose normalized values differ between bureaus."""

    union_fields = set()
    for bureau in _BUREAU_KEYS:
        branch = bureaus.get(bureau)
        if isinstance(branch, Mapping):
            union_fields.update(branch.keys())

    result: Dict[str, Dict[str, MutableMapping[str, Any]]] = {}
    for field in sorted(union_fields):
        normalized: MutableMapping[str, Any] = {}
        raw: MutableMapping[str, Any] = {}
        distinct = set()
        all_missing = True

        for bureau in _BUREAU_KEYS:
            branch = bureaus.get(bureau)
            value = branch.get(field) if isinstance(branch, Mapping) else None
            raw[bureau] = value
            norm_value = _normalize_value(field, value)
            normalized[bureau] = norm_value
            if norm_value is not None:
                all_missing = False
                distinct.add(norm_value)
            else:
                distinct.add(None)

        if all_missing:
            continue

        # Remove the placeholder None when other values exist to check actual disagreement.
        if None in distinct and len(distinct) > 1:
            distinct.remove(None)

        if len(distinct) > 1:
            result[field] = {"normalized": dict(normalized), "raw": dict(raw)}

    return result
