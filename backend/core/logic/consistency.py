"""Utilities for cross-bureau field consistency detection."""

from __future__ import annotations

import re
from typing import Any, Dict, Mapping, MutableMapping, Sequence

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
_HISTORY_FIELDS = {"two_year_payment_history", "seven_year_history"}

_AMOUNT_SANITIZE_RE = re.compile(r"[,$\s]")
_AMOUNT_RE = re.compile(r"-?\d+(?:\.\d+)?")
_HISTORY_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _is_missing(value: Any) -> bool:
    try:
        if value in _MISSING_SENTINELS:
            return True
    except TypeError:
        # Unhashable container values are not considered missing by membership.
        pass
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
    if field == "two_year_payment_history":
        return _normalize_two_year_history(value)
    if field == "seven_year_history":
        return _normalize_seven_year_history(value)
    if field in _MONEY_FIELDS:
        return _normalize_money(value)
    if field in _DATE_FIELDS or field.endswith("_date"):
        return _normalize_date(value)
    return _normalize_text(value)


def _normalize_history_status(value: Any) -> str | None:
    if value is None or value in _MISSING_SENTINELS:
        return None
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if isinstance(value, float) and value.is_integer():
            return str(int(value))
        return str(value)

    text = str(value).strip()
    if not text:
        return None
    normalized = text.upper().replace(" ", "")
    return normalized or None


def _flatten_history_values(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, Mapping):
        # Prefer well-known container keys when present.
        for key in ("values", "statuses", "history", "entries", "items"):
            if key in value:
                return _flatten_history_values(value.get(key))
        flattened: list[Any] = []
        for key in sorted(value.keys()):
            flattened.extend(_flatten_history_values(value[key]))
        return flattened
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        flattened: list[Any] = []
        for entry in value:
            if isinstance(entry, Mapping):
                if "status" in entry:
                    flattened.append(entry.get("status"))
                    continue
                if "value" in entry:
                    flattened.append(entry.get("value"))
                    continue
            flattened.extend(_flatten_history_values(entry))
        return flattened
    if isinstance(value, str):
        tokens = [token.strip() for token in _HISTORY_TOKEN_RE.findall(value.upper()) if token.strip()]
        if tokens:
            return tokens
        text = value.strip()
        return [text] if text else []
    return [value]


def _normalize_two_year_history(value: Any) -> tuple[str, ...] | None:
    if _is_missing(value):
        return None
    tokens = []
    for item in _flatten_history_values(value):
        token = _normalize_history_status(item)
        if token:
            tokens.append(token)
    if not tokens:
        return None
    return tuple(tokens)


def _normalize_history_count(value: Any) -> int | float | None:
    if value is None or value in _MISSING_SENTINELS:
        return None
    if isinstance(value, bool):
        return 1 if value else 0
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if isinstance(value, float) and value.is_integer():
            return int(value)
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    match = _AMOUNT_RE.search(text)
    if not match:
        return None
    number_text = match.group()
    try:
        number = float(number_text)
    except ValueError:
        return None
    if number.is_integer():
        return int(number)
    return number


def _normalize_seven_year_history(value: Any) -> Any:
    if _is_missing(value):
        return None
    if isinstance(value, Mapping):
        normalized_items = []
        for key in sorted(value.keys()):
            norm_key = _normalize_history_status(key) or ""
            norm_value = _normalize_history_count(value[key])
            normalized_items.append((norm_key, norm_value))
        if not normalized_items:
            return None
        return tuple(normalized_items)

    # Fall back to treating it like a sequence of status tokens.
    tokens = _normalize_two_year_history(value)
    if not tokens:
        return None
    return tokens


def compute_inconsistent_fields(bureaus: Mapping[str, Mapping[str, Any]]) -> Dict[str, Dict[str, MutableMapping[str, Any]]]:
    """Return fields whose normalized values differ between bureaus."""

    union_fields = set()
    for bureau in _BUREAU_KEYS:
        branch = bureaus.get(bureau)
        if isinstance(branch, Mapping):
            union_fields.update(branch.keys())
    for history_field in _HISTORY_FIELDS:
        history_blob = bureaus.get(history_field)
        if _is_missing(history_blob):
            continue
        union_fields.add(history_field)

    result: Dict[str, Dict[str, MutableMapping[str, Any]]] = {}
    for field in sorted(union_fields):
        normalized: MutableMapping[str, Any] = {}
        raw: MutableMapping[str, Any] = {}
        distinct = set()
        all_missing = True

        for bureau in _BUREAU_KEYS:
            if field in _HISTORY_FIELDS:
                history_blob = bureaus.get(field)
                if isinstance(history_blob, Mapping):
                    value = history_blob.get(bureau)
                else:
                    value = history_blob
            else:
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
