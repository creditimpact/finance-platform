"""Normalization and field consistency helpers for bureau data."""

from __future__ import annotations

import datetime as _dt
import re
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional, Sequence, Tuple

__all__ = ["compute_field_consistency", "compute_inconsistent_fields"]


_BUREAU_KEYS: Tuple[str, ...] = ("transunion", "experian", "equifax")
_MISSING_SENTINELS = {None, "", "--"}

_AMOUNT_FIELD_HINTS = {
    "amount",
    "balance",
    "limit",
    "payment",
    "value",
    "due",
    "credit",
    "loan",
    "debt",
}

_ENUM_DOMAINS: Dict[str, Dict[str, str]] = {
    "account_status": {
        "open": "open",
        "opened": "open",
        "active": "open",
        "current": "open",
        "close": "closed",
        "closed": "closed",
        "paidandclosed": "closed",
        "paidclosed": "closed",
        "closedpaid": "closed",
        "paid": "paid",
        "pif": "paid",
        "chargeoff": "chargeoff",
        "chargedoff": "chargeoff",
        "chargeoffsold": "chargeoff",
        "collection": "collection",
        "collections": "collection",
        "repossession": "repossession",
        "transferred": "transferred",
        "sold": "sold",
        "foreclosure": "foreclosure",
    },
    "payment_status": {
        "current": "ok",
        "paysasagreed": "ok",
        "payingasagreed": "ok",
        "paidasagreed": "ok",
        "ok": "ok",
        "neverlate": "ok",
        "chargeoff": "chargeoff",
        "chargedoff": "chargeoff",
        "collection": "collection",
        "collections": "collection",
        "repossession": "repossession",
        "late30": "late30",
        "late60": "late60",
        "late90": "late90",
        "late120": "late120",
        "late150": "late150",
        "late180": "late180",
        "120": "late120",
        "150": "late150",
        "180": "late180",
        "30": "late30",
        "60": "late60",
        "90": "late90",
    },
    "account_type": {
        "creditcard": "credit_card",
        "creditcards": "credit_card",
        "bankcreditcard": "bank_credit_card",
        "bankcreditcards": "bank_credit_card",
        "autoloan": "auto_loan",
        "autoloans": "auto_loan",
        "studentloan": "student_loan",
        "studentloans": "student_loan",
        "personalloan": "personal_loan",
        "personalloans": "personal_loan",
        "mortgage": "mortgage",
        "revolving": "revolving",
        "installment": "installment",
        "collection": "collection",
        "chargeaccount": "charge_account",
    },
    "creditor_type": {
        "bank": "bank",
        "allbanks": "bank",
        "bankcreditcards": "bank_credit_cards",
        "bankcard": "bank_credit_cards",
        "bankcards": "bank_credit_cards",
        "collectionagency": "collection_agency",
        "collectionagencies": "collection_agency",
        "creditunion": "credit_union",
        "mortgagelender": "mortgage_lender",
    },
    "dispute_status": {
        "disputed": "disputed",
        "indispute": "disputed",
        "dispute": "disputed",
        "open_dispute": "disputed",
        "notdisputed": "not_disputed",
        "nodispute": "not_disputed",
        "undisputed": "not_disputed",
        "resolved": "resolved",
        "closed": "resolved",
        "previouslydisputed": "previously_disputed",
    },
}

_HISTORY_FIELDS = {"two_year_payment_history", "seven_year_history"}
_ACCOUNT_NUMBER_FIELDS = {"account_number_display"}

_DATE_FORMATS = (
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%Y.%m.%d",
    "%m/%d/%Y",
    "%m-%d-%Y",
    "%d.%m.%Y",
    "%d/%m/%Y",
    "%m.%d.%Y",
    "%d-%m-%Y",
    "%m/%d/%y",
    "%d/%m/%y",
    "%Y%m%d",
)

_NON_ALNUM_RE = re.compile(r"[^A-Za-z0-9]")


def _is_missing(raw: Any) -> bool:
    try:
        if raw in _MISSING_SENTINELS:
            return True
    except TypeError:
        pass
    if isinstance(raw, str) and not raw.strip():
        return True
    return False


def normalize_amount(raw: Optional[str]) -> Optional[float]:
    """Parse $, commas, parentheses and sentinel markers into a float."""

    if _is_missing(raw):
        return None
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        return float(raw)

    text = str(raw).strip()
    if not text:
        return None

    negative = False
    if text.startswith("(") and text.endswith(")"):
        negative = True
        text = text[1:-1]

    cleaned = re.sub(r"[,$\s]", "", text)
    if not cleaned:
        return None

    match = re.search(r"-?\d+(?:\.\d+)?", cleaned)
    if not match:
        return None

    try:
        value = float(match.group())
    except ValueError:
        return None

    if negative and value >= 0:
        value = -value
    return value


def normalize_date(raw: Optional[str]) -> Optional[str]:
    """Normalize several common date formats to YYYY-MM-DD."""

    if _is_missing(raw):
        return None
    text = str(raw).strip()
    if not text:
        return None

    cleaned = text.replace("\\", "/").replace("\u2013", "-").replace("\u2014", "-")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    for fmt in _DATE_FORMATS:
        try:
            parsed = _dt.datetime.strptime(cleaned, fmt)
        except ValueError:
            continue
        return parsed.date().isoformat()

    digits = re.findall(r"\d+", cleaned)
    if len(digits) == 3:
        first, second, third = digits
        year: Optional[int]
        month: Optional[int]
        day: Optional[int]

        if len(first) == 4:
            year, month, day = int(first), int(second), int(third)
        elif len(third) == 4:
            year = int(third)
            month = int(first)
            day = int(second)
            if month > 12 and day <= 12:
                month, day = day, month
        else:
            return None

        try:
            parsed = _dt.date(year, month, day)
        except ValueError:
            return None
        return parsed.isoformat()

    return None


def _normalize_text(raw: Any) -> Optional[str]:
    if _is_missing(raw):
        return None
    text = str(raw).strip().lower()
    if not text:
        return None
    cleaned = re.sub(r"[^a-z0-9]+", " ", text)
    collapsed = " ".join(cleaned.split())
    return collapsed or None


def normalize_enum(field: str, raw: Optional[str]) -> Optional[str]:
    """Normalize enums using domain-specific alias tables."""

    if _is_missing(raw):
        return None

    text = str(raw).strip().lower()
    if not text:
        return None

    compact = _NON_ALNUM_RE.sub("", text)
    domain = _ENUM_DOMAINS.get(field, {})
    if compact in domain:
        return domain[compact]
    if text in domain:
        return domain[text]
    return " ".join(text.split())


def normalize_account_number_display(raw: Optional[str]) -> Dict[str, Optional[str]]:
    """Return a display/last4 structure for account numbers."""

    if _is_missing(raw):
        return {"display": "", "last4": None}

    text = str(raw).strip()
    digits = re.findall(r"\d", text)
    last4 = "".join(digits[-4:]) if digits else None
    return {"display": text, "last4": last4 if len(last4 or "") == 4 else None}


def _history_items(raw: Any) -> Iterable[Any]:
    if raw is None:
        return []
    if isinstance(raw, Mapping):
        for key in ("tokens", "values", "statuses", "history", "entries", "items"):
            if key in raw:
                return _history_items(raw[key])
        items: list[Any] = []
        for value in raw.values():
            items.extend(_history_items(value))
        return items
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
        items: list[Any] = []
        for value in raw:
            items.extend(_history_items(value))
        return items
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return []
        tokens = re.findall(r"[A-Za-z0-9]+", text)
        if tokens:
            return tokens
        return [text]
    return [raw]


def _canonical_history_token(token: Any) -> Optional[str]:
    if _is_missing(token):
        return None
    text = str(token).strip().upper()
    if not text:
        return None
    cleaned = re.sub(r"[^A-Z0-9]", "", text)
    if not cleaned:
        return None

    mapping = {
        "OK": "OK",
        "CUR": "OK",
        "CURRENT": "OK",
        "C": "OK",
        "CO": "CO",
        "COCOUNT": "CO",
        "CHARGEOFF": "CO",
        "CHARGE": "CO",
        "CHARGEDOFF": "CO",
        "CHARGEOFFS": "CO",
        "CHARGEOFFCOUNT": "CO",
        "DELINQ": "DELINQ",
        "LATE": "LATE",
        "LATE30": "30",
        "PAST30": "30",
        "PAST30DAYS": "30",
        "PASTDUE30": "30",
        "30": "30",
        "030": "30",
        "30DAY": "30",
        "30DAYS": "30",
        "30DAYSLATE": "30",
        "LATE60": "60",
        "PAST60": "60",
        "PAST60DAYS": "60",
        "PASTDUE60": "60",
        "60": "60",
        "060": "60",
        "60DAY": "60",
        "60DAYSLATE": "60",
        "LATE90": "90",
        "PAST90": "90",
        "PAST90DAYS": "90",
        "PASTDUE90": "90",
        "90": "90",
        "90DAY": "90",
        "90DAYSLATE": "90",
        "PAST120": "120",
        "PASTDUE120": "120",
        "120": "120",
        "PAST150": "150",
        "150": "150",
        "PAST180": "180",
        "180": "180",
    }

    return mapping.get(cleaned, cleaned)


def normalize_two_year_history(raw: Any) -> Dict[str, Any]:
    """Normalize two-year payment history tokens and summary counts."""

    tokens: list[str] = []
    counts = {"CO": 0, "late30": 0, "late60": 0, "late90": 0}

    for item in _history_items(raw):
        canon = _canonical_history_token(item)
        if not canon:
            continue
        tokens.append(canon)
        if canon == "CO":
            counts["CO"] += 1
        elif canon in {"30"}:
            counts["late30"] += 1
        elif canon in {"60"}:
            counts["late60"] += 1
        elif canon in {"90", "120", "150", "180"}:
            counts["late90"] += 1

    return {"tokens": tokens, "counts": counts}


def _parse_int(value: Any) -> int:
    if _is_missing(value):
        return 0
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return int(value)
    text = str(value).strip()
    if not text:
        return 0
    match = re.search(r"-?\d+", text)
    if not match:
        return 0
    return int(match.group())


def normalize_seven_year_history(raw: Any) -> Dict[str, int]:
    """Normalize seven-year history counters to late30/late60/late90."""

    result = {"late30": 0, "late60": 0, "late90": 0}
    if _is_missing(raw):
        return result

    def consume_entry(key: Any, value: Any) -> None:
        token = _canonical_history_token(key)
        if not token:
            return
        if token in {"30", "LATE30"}:
            result["late30"] += _parse_int(value)
        elif token in {"60", "LATE60"}:
            result["late60"] += _parse_int(value)
        elif token in {"90", "120", "150", "180", "LATE90", "CO"}:
            result["late90"] += _parse_int(value)

    if isinstance(raw, Mapping):
        for key, value in raw.items():
            consume_entry(key, value)
    elif isinstance(raw, str):
        for token in _history_items(raw):
            consume_entry(token, 1)
    elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
        for entry in raw:
            if isinstance(entry, Mapping) and "type" in entry and "value" in entry:
                consume_entry(entry.get("type"), entry.get("value"))
            else:
                for token in _history_items(entry):
                    consume_entry(token, 1)
    else:
        consume_entry("late30", raw)

    return result


def _looks_like_amount(field: str) -> bool:
    lowered = field.lower()
    if any(lowered.endswith(suffix) for suffix in _AMOUNT_FIELD_HINTS):
        return True
    return any(token in lowered for token in ("amount", "balance", "limit", "payment"))


def _normalize_field(field: str, value: Any) -> Any:
    if field in _HISTORY_FIELDS:
        if field == "two_year_payment_history":
            return normalize_two_year_history(value)
        return normalize_seven_year_history(value)
    if field in _ACCOUNT_NUMBER_FIELDS:
        return normalize_account_number_display(value)
    if field in _ENUM_DOMAINS:
        return normalize_enum(field, value)
    if "date" in field.lower():
        return normalize_date(value)
    if _looks_like_amount(field):
        return normalize_amount(value)
    return _normalize_text(value)


def _freeze_value(field: str, value: Any) -> Any:
    if value is None:
        return None
    if field == "account_number_display":
        if isinstance(value, Mapping):
            return value.get("last4") or value.get("display")
    if field == "two_year_payment_history":
        if isinstance(value, Mapping):
            counts = value.get("counts", {})
            tokens = tuple(value.get("tokens", []))
            return (
                counts.get("CO", 0),
                counts.get("late30", 0),
                counts.get("late60", 0),
                counts.get("late90", 0),
                tokens,
            )
    if field == "seven_year_history":
        if isinstance(value, Mapping):
            return (
                value.get("late30", 0),
                value.get("late60", 0),
                value.get("late90", 0),
            )
    if isinstance(value, Mapping):
        return tuple(sorted((key, _freeze_value(field, item)) for key, item in value.items()))
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(_freeze_value(field, item) for item in value)
    return value


def _extract_value(bureaus: Mapping[str, Any], bureau: str, field: str) -> Any:
    if field in _HISTORY_FIELDS:
        blob = bureaus.get(field)
        if isinstance(blob, Mapping) and bureau in blob:
            return blob[bureau]
    branch = bureaus.get(bureau)
    if isinstance(branch, Mapping):
        return branch.get(field)
    return None


def _determine_consensus(field: str, groups: Mapping[Any, Sequence[str]]) -> Tuple[str, list[str]]:
    items = [(key, list(bureaus)) for key, bureaus in groups.items() if bureaus]
    if not items:
        return "unanimous", []

    if len(items) == 1:
        return "unanimous", []

    total = sum(len(bureaus) for _, bureaus in items)
    sorted_items = sorted(items, key=lambda item: (-len(item[1]), str(item[0])))
    top_key, top_bureaus = sorted_items[0]
    top_count = len(top_bureaus)

    if top_count > total / 2:
        disagreeing = [bureau for key, bureaus in items if key != top_key for bureau in bureaus]
        disagreeing.sort()
        return "majority", disagreeing

    disagreeing = [bureau for _, bureaus in items for bureau in bureaus]
    disagreeing.sort()
    return "split", disagreeing


def compute_field_consistency(bureaus_json: Dict[str, Any]) -> Dict[str, Any]:
    """Compute normalized values and consensus for every field across bureaus."""

    fields = set(_HISTORY_FIELDS)

    for bureau in _BUREAU_KEYS:
        branch = bureaus_json.get(bureau)
        if isinstance(branch, Mapping):
            fields.update(branch.keys())

    results: Dict[str, Any] = {}

    for field in sorted(fields):
        normalized: MutableMapping[str, Any] = {}
        raw: MutableMapping[str, Any] = {}
        groups: Dict[Any, list[str]] = {}
        missing_bureaus: list[str] = []

        for bureau in _BUREAU_KEYS:
            value = _extract_value(bureaus_json, bureau, field)
            raw[bureau] = value
            norm_value = _normalize_field(field, value)
            normalized[bureau] = norm_value
            if norm_value is None:
                missing_bureaus.append(bureau)
            key = _freeze_value(field, norm_value)
            groups.setdefault(key, []).append(bureau)

        if all(norm is None for norm in normalized.values()):
            continue

        consensus, disagreeing = _determine_consensus(field, groups)
        results[field] = {
            "consensus": consensus,
            "normalized": dict(normalized),
            "raw": dict(raw),
            "disagreeing_bureaus": disagreeing,
            "missing_bureaus": sorted({bureau for bureau in missing_bureaus}),
        }

    return results


def compute_inconsistent_fields(bureaus_json: Mapping[str, Any]) -> Dict[str, Any]:
    """Return fields whose consensus is not unanimous."""

    consistency = compute_field_consistency(dict(bureaus_json))
    return {
        field: info
        for field, info in consistency.items()
        if info.get("consensus") != "unanimous"
    }

