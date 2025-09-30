"""Utilities for cross-bureau field consistency detection."""

from __future__ import annotations

import datetime as _dt
import re
import string
from typing import Any, Dict, Mapping, MutableMapping, Sequence, Tuple

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
_REMARK_FIELDS = {"remarks", "creditor_remarks"}
_ACCOUNT_NUMBER_FIELDS = {
    "account_number_display",
    "account_number",
    "account_number_masked",
}

_ACCOUNT_STATUS_ALIASES = {
    "open": "open",
    "opened": "open",
    "opn": "open",
    "close": "closed",
    "closed": "closed",
    "paidclosed": "closed",
    "paidandclosed": "closed",
    "chargeoff": "chargeoff",
    "chargedoff": "chargeoff",
    "chargeofftransferred": "chargeoff",
    "chargeoffsold": "chargeoff",
    "collection": "collection",
    "collections": "collection",
    "incollection": "collection",
    "incollections": "collection",
    "repossession": "repossession",
}

_PAYMENT_STATUS_ALIASES = {
    "current": "ok",
    "paid": "ok",
    "paysasagreed": "ok",
    "ok": "ok",
    "neverlate": "ok",
    "paysasagreed": "ok",
    "payingasagreed": "ok",
    "paidasagreed": "ok",
    "chargeoff": "chargeoff",
    "chargedoff": "chargeoff",
    "collection": "collection",
    "collections": "collection",
    "collectionaccount": "collection",
    "repossession": "repossession",
}

_ACCOUNT_TYPE_ALIASES = {
    "creditcard": "credit_card",
    "creditcards": "credit_card",
    "bankcreditcard": "bank_credit_cards",
    "bankcreditcards": "bank_credit_cards",
    "autoloan": "auto_loan",
    "autoloans": "auto_loan",
    "studentloan": "student_loan",
    "studentloans": "student_loan",
    "personalloan": "personal_loan",
    "personalloans": "personal_loan",
}

_CREDITOR_TYPE_ALIASES = {
    "bankcreditcards": "bank_credit_cards",
    "bank": "bank",
    "allbanks": "all_banks",
    "bankcards": "bank_credit_cards",
    "collectionagency": "collection_agency",
    "collectionagencies": "collection_agency",
}

_DATE_FORMATS = (
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%Y.%m.%d",
    "%m/%d/%Y",
    "%m-%d-%Y",
    "%m.%d.%Y",
    "%d/%m/%Y",
    "%d-%m-%Y",
    "%d.%m.%Y",
    "%m/%d/%y",
    "%d/%m/%y",
    "%Y%m%d",
)

_RE_PUNCT = re.compile(rf"[{re.escape(string.punctuation)}]")

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
    if _is_missing(value):
        return None
    text = str(value).strip()
    if not text:
        return None

    cleaned = text.replace(",", " ").replace("\u2013", "-").replace("\u2014", "-")
    cleaned = cleaned.replace("\\", "/")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    for fmt in _DATE_FORMATS:
        try:
            parsed = _dt.datetime.strptime(cleaned, fmt)
        except ValueError:
            continue
        return parsed.date().isoformat()

    digits = [part for part in re.findall(r"\d+", cleaned)]
    if len(digits) == 3:
        first, second, third = digits
        if len(first) == 4:
            year, month, day = first, second, third
        elif len(third) == 4:
            day_candidate = int(first)
            month_candidate = int(second)
            if day_candidate > 12 and month_candidate <= 12:
                day, month, year = first, second, third
            elif month_candidate > 12 and day_candidate <= 12:
                month, day, year = first, second, third
            else:
                month, day, year = first, second, third
        else:
            return None

        try:
            parsed = _dt.date(int(year), int(month), int(day))
        except ValueError:
            return None
        return parsed.isoformat()

    return None


def _normalize_compact_token(value: Any) -> str | None:
    if _is_missing(value):
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    token = re.sub(r"[^a-z0-9]+", "", text)
    return token or None


def _normalize_words_token(value: Any) -> str | None:
    if _is_missing(value):
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    collapsed = re.sub(r"[^a-z0-9]+", " ", text)
    tokens = [part for part in collapsed.split() if part]
    if not tokens:
        return None
    return "_".join(tokens)


def _normalize_account_status(value: Any) -> str | None:
    token = _normalize_compact_token(value)
    if not token:
        return None
    return _ACCOUNT_STATUS_ALIASES.get(token, token)


def _normalize_payment_status(value: Any) -> str | None:
    compact = _normalize_compact_token(value)
    text = str(value).lower() if not _is_missing(value) else ""
    if not compact and not text:
        return None
    if compact in _PAYMENT_STATUS_ALIASES:
        return _PAYMENT_STATUS_ALIASES[compact]
    if "charge" in text and "off" in text:
        return "chargeoff"
    if "collection" in text:
        return "collection"
    match = re.search(r"(\d{1,3})", text)
    if match:
        number = match.group(1)
        if "late" in text or number in {"30", "60", "90", "120", "150", "180"}:
            mapping = {
                "30": "late30",
                "60": "late60",
                "90": "late90",
                "120": "late120",
                "150": "late150",
                "180": "late180",
            }
            return mapping.get(number, f"late{number}")
    if any(word in text for word in ("current", "ok", "pays", "paying", "paid")):
        return "ok"
    return compact or None


def _normalize_account_type(value: Any) -> str | None:
    token = _normalize_words_token(value)
    if not token:
        return None
    alias_key = token.replace("_", "")
    return _ACCOUNT_TYPE_ALIASES.get(alias_key, token)


def _normalize_creditor_type(value: Any) -> str | None:
    token = _normalize_words_token(value)
    if not token:
        return None
    alias_key = token.replace("_", "")
    return _CREDITOR_TYPE_ALIASES.get(alias_key, token)


def _normalize_remark_text(value: Any) -> str | None:
    if _is_missing(value):
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    text = _RE_PUNCT.sub(" ", text)
    tokens = [token for token in text.split() if token]
    if not tokens:
        return None
    return " ".join(tokens)


def _normalize_account_number(value: Any) -> Mapping[str, str] | None:
    if _is_missing(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    digits = re.findall(r"\d", text)
    if not digits:
        return None
    last4 = "".join(digits[-4:])
    if not last4:
        return None
    masked = re.sub(r"\s+", "", text)
    return {"last4": last4, "display": masked}


def _normalize_value(field: str, value: Any) -> Any:
    if field == "two_year_payment_history":
        return _normalize_two_year_history(value)
    if field == "seven_year_history":
        return _normalize_seven_year_history(value)
    if field in _ACCOUNT_NUMBER_FIELDS:
        return _normalize_account_number(value)
    if field == "account_status":
        return _normalize_account_status(value)
    if field == "payment_status":
        return _normalize_payment_status(value)
    if field == "account_type":
        return _normalize_account_type(value)
    if field == "creditor_type":
        return _normalize_creditor_type(value)
    if field in _MONEY_FIELDS:
        return _normalize_money(value)
    if field in _DATE_FIELDS or field.endswith("_date"):
        return _normalize_date(value)
    if field in _REMARK_FIELDS:
        return _normalize_remark_text(value)
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


def _normalize_two_year_history(value: Any) -> Mapping[str, Any] | None:
    if _is_missing(value):
        return None
    tokens = []
    for item in _flatten_history_values(value):
        token = _normalize_history_status(item)
        if token:
            tokens.append(token)
    if not tokens:
        return None
    canonical_tokens: list[str] = []
    summary_counts = {"co_count": 0, "late30": 0, "late60": 0, "late90": 0}
    for token in tokens:
        canon = _canonicalize_history_code(token)
        canonical_tokens.append(canon)
        if canon == "CO":
            summary_counts["co_count"] += 1
        elif canon in {"30", "60", "90", "120", "150", "180"}:
            if canon == "30":
                summary_counts["late30"] += 1
            elif canon == "60":
                summary_counts["late60"] += 1
            else:
                summary_counts["late90"] += 1
    return {"codes": tuple(canonical_tokens), "summary": summary_counts}


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
        normalized: Dict[str, Any] = {}
        for key in value.keys():
            norm_key = (_normalize_history_status(key) or "").lower()
            if not norm_key:
                continue
            norm_value = _normalize_history_count(value[key])
            normalized[norm_key] = norm_value
        if not normalized:
            return None
        return normalized

    # Fall back to treating it like a sequence of status tokens.
    tokens = _normalize_two_year_history(value)
    if not tokens:
        return None
    return tokens


def _canonicalize_history_code(token: str) -> str:
    token_upper = token.upper()
    if token_upper in {"OK", "C", "CUR", "CURR", "CURRENT", "0"}:
        return "OK"
    if token_upper in {"CO", "C/O", "CHARGEOFF", "CHGOFF", "CHARGE-OFF"}:
        return "CO"
    match = re.search(r"(\d{1,3})", token_upper)
    if match:
        return match.group(1)
    return token_upper


def _freeze_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        keys = {str(k) for k in value.keys()}
        if "last4" in keys and keys.issubset({"last4", "display"}):
            return ("acct_last4", value.get("last4"))
        return tuple(sorted((str(k), _freeze_value(v)) for k, v in value.items()))
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(_freeze_value(item) for item in value)
    return value


def _build_field_consistency(
    bureaus: Mapping[str, Mapping[str, Any]]
) -> Dict[str, Dict[str, Any]]:
    union_fields = set()
    for bureau in _BUREAU_KEYS:
        branch = bureaus.get(bureau)
        if isinstance(branch, Mapping):
            union_fields.update(branch.keys())
    union_fields.update(_HISTORY_FIELDS)

    details: Dict[str, Dict[str, Any]] = {}
    for field in sorted(union_fields):
        normalized: MutableMapping[str, Any] = {}
        raw: MutableMapping[str, Any] = {}
        value_groups: Dict[Any, list[str]] = {}
        all_missing = True

        for bureau in _BUREAU_KEYS:
            if field in _HISTORY_FIELDS:
                history_blob = bureaus.get(field)
                if isinstance(history_blob, Mapping):
                    value = history_blob.get(bureau)
                else:
                    value = history_blob
                if value is None:
                    branch = bureaus.get(bureau)
                    if isinstance(branch, Mapping):
                        value = branch.get(field)
            else:
                branch = bureaus.get(bureau)
                value = branch.get(field) if isinstance(branch, Mapping) else None
            raw[bureau] = value
            norm_value = _normalize_value(field, value)
            normalized[bureau] = norm_value
            if norm_value is not None:
                all_missing = False
            frozen = _freeze_value(norm_value)
            value_groups.setdefault(frozen, []).append(bureau)

        if all_missing:
            continue

        consensus, disagreeing = _determine_consensus(value_groups)
        details[field] = {
            "consensus": consensus,
            "normalized": dict(normalized),
            "raw": dict(raw),
            "disagreeing_bureaus": disagreeing,
        }

    return details


def _determine_consensus(value_groups: Mapping[Any, Sequence[str]]) -> Tuple[str, list[str]]:
    items = [
        (key, list(bureaus)) for key, bureaus in value_groups.items() if bureaus
    ]
    if not items:
        return "unanimous", []

    total = sum(len(b) for _, b in items)
    sorted_items = sorted(items, key=lambda item: (-len(item[1]), str(item[0])))
    top_key, top_bureaus = sorted_items[0]
    top_count = len(top_bureaus)
    unique_keys = {key for key, _ in items}
    frozen_none = _freeze_value(None)
    non_missing_keys = {key for key in unique_keys if key != frozen_none}

    if len(unique_keys) == 1:
        return "unanimous", []
    if non_missing_keys and len(non_missing_keys) == 1 and len(unique_keys) == 2 and frozen_none in unique_keys:
        return "unanimous", []
    if top_count > total / 2:
        consensus = "majority"
    else:
        consensus = "split"

    disagreeing = []
    for key, bureaus in items:
        if key == top_key:
            continue
        disagreeing.extend(bureaus)
    disagreeing.sort()
    return consensus, disagreeing


def compute_inconsistent_fields(bureaus: Mapping[str, Mapping[str, Any]]) -> Dict[str, Dict[str, MutableMapping[str, Any]]]:
    """Return fields whose normalized values differ between bureaus."""
    details = _build_field_consistency(bureaus)
    result: Dict[str, Dict[str, MutableMapping[str, Any]]] = {}
    for field, info in details.items():
        if info.get("consensus") == "unanimous":
            continue
        result[field] = {
            "normalized": info.get("normalized", {}),
            "raw": info.get("raw", {}),
            "consensus": info.get("consensus"),
            "disagreeing_bureaus": info.get("disagreeing_bureaus", []),
        }
    return result


def compute_field_consistency(bureaus: Mapping[str, Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Return normalized field comparisons for all available fields."""

    return _build_field_consistency(bureaus)


__all__.append("compute_field_consistency")
