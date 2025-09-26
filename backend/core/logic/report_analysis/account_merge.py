"""Utilities for deterministic 0–100 merge scoring and tagging."""

from __future__ import annotations

import json
import logging
import os
import re
from copy import deepcopy
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Mapping, Optional, Set, Tuple, Union

from backend import config as app_config
from backend.core.io.tags import read_tags, upsert_tag, write_tags_atomic
from backend.core.logic.normalize.accounts import normalize_acctnum as _normalize_acctnum_basic
from backend.core.merge import acctnum
from backend.core.merge.acctnum import acctnum_level, normalize_level
from backend.core.logic.report_analysis.ai_pack import build_ai_pack_for_pair
from backend.core.logic.report_analysis import config as merge_config

__all__ = [
    "load_bureaus",
    "get_merge_cfg",
    "gen_unordered_pairs",
    "score_pair_0_100",
    "score_all_pairs_0_100",
    "choose_best_partner",
    "persist_merge_tags",
    "score_and_tag_best_partners",
    "merge_v2_only_enabled",
    "build_merge_pair_tag",
    "build_merge_best_tag",
]


logger = logging.getLogger(__name__)


POINTS_ACCTNUM_VISIBLE = 28


def _coerce_positive_int(value: Optional[int]) -> Optional[int]:
    """Return ``value`` when it is a positive integer, otherwise ``None``."""

    if value is None:
        return None
    try:
        coerced = int(value)
    except (TypeError, ValueError):
        return None
    return coerced if coerced > 0 else None


def _read_candidate_limits(env: Optional[Mapping[str, str]] = None) -> Tuple[Optional[int], Optional[int]]:
    """Return global and per-account candidate limits from the environment."""

    env_mapping: Mapping[str, str]
    if env is None:
        env_mapping = os.environ
    else:
        env_mapping = env

    global_limit = _coerce_positive_int(env_mapping.get("MERGE_CANDIDATE_LIMIT"))
    if global_limit is None:
        legacy_limit = env_mapping.get("MERGE_PAIR_LIMIT")
        global_limit = _coerce_positive_int(legacy_limit)

    per_account_limit = _coerce_positive_int(env_mapping.get("MAX_CANDIDATES_PER_ACCOUNT"))

    return global_limit, per_account_limit


def _sanitize_acct_level(value: Any) -> str:
    """Return the supported account-number level for arbitrary input."""

    if value is None:
        candidate: str | None = None
    elif isinstance(value, str):
        candidate = value
    else:
        candidate = str(value)
    return normalize_level(candidate)


def _priority_label_for_level(level: str) -> Tuple[int, int, str]:
    """Return (category, subscore, label) for the provided account-number level."""

    normalized = _sanitize_acct_level(level)
    if normalized == "exact_or_known_match":
        return (0, 1, "hard:acctnum_visible_digits")
    return (3, 0, "default")


def _priority_category(level: str, dates_all: bool, score_gate: bool) -> Tuple[int, int, str]:
    """Return the ordered priority bucket metadata for a pair."""

    category, subscore, label = _priority_label_for_level(level)
    if category == 0:
        return category, subscore, label
    if dates_all:
        return 1, 0, "dates_all"
    if score_gate:
        return 2, 0, "score_gate"
    return category, subscore, label


def is_missing(value: Any) -> bool:
    """Return True when a value represents an explicit missing sentinel."""

    return value in {None, "", "--"}


def _read_env_int(env: Mapping[str, str], key: str, default: int) -> int:
    raw = env.get(key)
    if raw is None:
        return int(default)
    try:
        return int(raw)
    except (TypeError, ValueError):
        return int(default)


def _read_env_float(env: Mapping[str, str], key: str, default: float) -> float:
    raw = env.get(key)
    if raw is None:
        return float(default)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return float(default)


def _read_env_flag(env: Mapping[str, str], key: str, default: bool) -> bool:
    raw = env.get(key)
    if raw is None:
        return bool(default)
    lowered = str(raw).strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    return bool(default)


def merge_v2_only_enabled() -> bool:
    """Return True when legacy merge artefact writes must be skipped."""

    return merge_config.get_merge_v2_only()


def gen_unordered_pairs(indices: List[int]) -> List[Tuple[int, int]]:
    """Return all unordered pairs (i, j) with i < j and no duplicates."""

    unique = sorted(set(indices))
    return [(i, j) for pos, i in enumerate(unique) for j in unique[pos + 1 :]]


def load_bureaus(
    sid: str, idx: int, runs_root: Path = Path("runs")
) -> Dict[str, Dict[str, Any]]:
    """Load bureau data for a case account, normalizing missing values."""

    bureaus_path = runs_root / sid / "cases" / "accounts" / str(idx) / "bureaus.json"
    if not bureaus_path.exists():
        raise FileNotFoundError(
            f"bureaus.json not found for sid={sid!r} index={idx} under {runs_root}"
        )

    with bureaus_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, Mapping):
        logger.warning(
            "Unexpected bureaus payload type %s for sid=%s idx=%s; treating as empty",
            type(data).__name__,
            sid,
            idx,
        )
        data = {}

    result: Dict[str, Dict[str, Any]] = {}
    for bureau in ("transunion", "experian", "equifax"):
        branch = data.get(bureau) if isinstance(data, Mapping) else None
        if isinstance(branch, Mapping):
            cleaned = {
                key: value
                for key, value in branch.items()
                if not is_missing(value)
            }
        else:
            cleaned = {}
        result[bureau] = dict(cleaned)

    return result


@dataclass
class MergeCfg:
    """Centralized configuration for deterministic account merging."""

    points: Mapping[str, int]
    thresholds: Mapping[str, int]
    triggers: Mapping[str, Union[int, str, bool]]
    tolerances: Mapping[str, Union[int, float]]


_POINT_DEFAULTS: Dict[str, int] = {
    "balance_owed": 31,
    "account_number": POINTS_ACCTNUM_VISIBLE,
    "last_payment": 12,
    "past_due_amount": 8,
    "high_balance": 6,
    "creditor_type": 3,
    "account_type": 3,
    "payment_amount": 2,
    "credit_limit": 1,
    "last_verified": 1,
    "date_of_last_activity": 2,
    "date_reported": 1,
    "date_opened": 1,
    "closed_date": 1,
}

_THRESHOLD_DEFAULTS: Dict[str, int] = {
    "AI_THRESHOLD": 26,
    "AUTO_MERGE_THRESHOLD": 70,
}

_TRIGGER_DEFAULTS: Dict[str, Union[int, str, bool]] = {
    "MERGE_AI_ON_BALOWED_EXACT": 1,
    "MERGE_AI_ON_HARD_ACCTNUM": 1,
    "MERGE_AI_ON_MID_K": 26,
    "MERGE_AI_ON_ALL_DATES": 1,
}

_TOLERANCE_DEFAULTS: Dict[str, Union[int, float]] = {
    "AMOUNT_TOL_ABS": 50.0,
    "AMOUNT_TOL_RATIO": 0.01,
    "LAST_PAYMENT_DAY_TOL": 7,
    "COUNT_ZERO_PAYMENT_MATCH": 0,
}

def get_merge_cfg(env: Optional[Mapping[str, str]] = None) -> MergeCfg:
    """Return merge configuration using environment overrides when provided."""

    env_mapping: Mapping[str, str]
    if env is None:
        env_mapping = os.environ
    else:
        env_mapping = env

    points = dict(_POINT_DEFAULTS)

    thresholds = {
        key: _read_env_int(env_mapping, key, default)
        for key, default in _THRESHOLD_DEFAULTS.items()
    }

    triggers: Dict[str, Union[int, str, bool]] = {}
    triggers["MERGE_AI_ON_BALOWED_EXACT"] = _read_env_flag(
        env_mapping,
        "MERGE_AI_ON_BALOWED_EXACT",
        bool(_TRIGGER_DEFAULTS["MERGE_AI_ON_BALOWED_EXACT"]),
    )
    triggers["MERGE_AI_ON_HARD_ACCTNUM"] = _read_env_flag(
        env_mapping,
        "MERGE_AI_ON_HARD_ACCTNUM",
        bool(_TRIGGER_DEFAULTS["MERGE_AI_ON_HARD_ACCTNUM"]),
    )
    triggers["MERGE_AI_ON_MID_K"] = _read_env_int(
        env_mapping,
        "MERGE_AI_ON_MID_K",
        int(_TRIGGER_DEFAULTS["MERGE_AI_ON_MID_K"]),
    )
    triggers["MERGE_AI_ON_ALL_DATES"] = _read_env_flag(
        env_mapping,
        "MERGE_AI_ON_ALL_DATES",
        bool(_TRIGGER_DEFAULTS["MERGE_AI_ON_ALL_DATES"]),
    )

    tolerances = {
        key: (
            _read_env_int(env_mapping, key, int(default))
            if isinstance(default, int)
            else _read_env_float(env_mapping, key, float(default))
        )
        for key, default in _TOLERANCE_DEFAULTS.items()
    }

    # Ensure tolerance types remain float for ratio/absolute values
    if isinstance(tolerances["AMOUNT_TOL_ABS"], int):
        tolerances["AMOUNT_TOL_ABS"] = float(tolerances["AMOUNT_TOL_ABS"])
    if isinstance(tolerances["AMOUNT_TOL_RATIO"], int):
        tolerances["AMOUNT_TOL_RATIO"] = float(tolerances["AMOUNT_TOL_RATIO"])

    # Count-zero-payment match is an integer toggle but maintain numeric type explicitly.
    tolerances["COUNT_ZERO_PAYMENT_MATCH"] = int(
        tolerances["COUNT_ZERO_PAYMENT_MATCH"]
    )

    return MergeCfg(
        points=points,
        thresholds=thresholds,
        triggers=triggers,
        tolerances=tolerances,
    )


# ---------------------------------------------------------------------------
# Deterministic merge helpers
# ---------------------------------------------------------------------------

_AMOUNT_SANITIZE_RE = re.compile(r"[\s$,/]")
_AMOUNT_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")
_ACCOUNT_LEVEL_ORDER = {
    "none": 0,
    "exact_or_known_match": 1,
}
_ACCOUNT_NUMBER_WEIGHTS = {
    "exact_or_known_match": POINTS_ACCTNUM_VISIBLE,
}
_ACCOUNT_STRONG_LEVELS = {"exact_or_known_match"}
_MASK_CHARS = {"*", "x", "X", "•", "●"}

_ACCOUNT_LEVEL_PRIORITY = {
    "none": 0,
    "exact_or_known_match": 1,
}

_IDENTITY_FIELD_SET = {
    "account_number",
    "creditor_type",
    "date_opened",
    "closed_date",
    "date_of_last_activity",
    "date_reported",
    "last_verified",
}


def normalize_acctnum(raw: str | None) -> Dict[str, object]:
    """Return canonical metadata for account-number comparisons.

    Returns a dictionary with:

    ``raw``
        Original input string (or ``None``).
    ``digits``
        Concatenated digits extracted from the account number (may be empty).
    ``canon_mask``
        Canonical masked form after removing whitespace, dashes, dots, and
        collapsing mask runs.
    ``has_digits``
        Boolean indicating whether any digits were observed.
    ``has_mask``
        Boolean indicating whether any masking characters were present.
    ``visible_digits``
        Count of visible digits observed after removing masks and spacing.
    """

    raw_value = raw if isinstance(raw, str) else ("" if raw is None else str(raw))
    if raw_value is None:
        raw_value = ""

    stripped = re.sub(r"[\s\-.]+", "", raw_value)
    if not stripped:
        return {
            "raw": raw,
            "digits": "",
            "canon_mask": "",
            "has_digits": False,
            "has_mask": False,
            "visible_digits": 0,
        }

    translated_chars: List[str] = []
    digits_chars: List[str] = []
    visible_digit_count = 0
    for char in stripped:
        if char.isdigit():
            digits_chars.append(char)
            translated_chars.append(char)
            visible_digit_count += 1
        elif char in _MASK_CHARS:
            translated_chars.append("*")
        else:
            translated_chars.append(char.upper())

    canon_mask = re.sub(r"\*+", "*", "".join(translated_chars))
    digits = "".join(digits_chars)
    has_digits = bool(digits)

    has_mask = "*" in canon_mask

    return {
        "raw": raw,
        "digits": digits,
        "canon_mask": canon_mask,
        "has_digits": has_digits,
        "has_mask": has_mask,
        "visible_digits": visible_digit_count,
    }


DIGITS_RE = re.compile(r"\d")


def _digits_only(raw: str | None) -> str:
    return "".join(DIGITS_RE.findall(raw or ""))


def acctnum_match_level(
    a_raw: str | None, b_raw: str | None
) -> Tuple[str, Dict[str, Dict[str, str]]]:
    """Return the visible-digits account-number level with debug payload."""

    a_value = str(a_raw) if a_raw is not None else ""
    b_value = str(b_raw) if b_raw is not None else ""
    a_digits = _digits_only(a_value)
    b_digits = _digits_only(b_value)

    level, detail = acctnum_level(a_value, b_value)

    debug: Dict[str, Dict[str, str]] = {
        "a": {
            "raw": a_value,
            "digits": a_digits,
        },
        "b": {
            "raw": b_value,
            "digits": b_digits,
        },
        "short": str(detail.get("short", "")),
        "long": str(detail.get("long", "")),
    }
    if "why" in detail:
        debug["why"] = str(detail["why"])

    return level, debug
_TYPE_ALIAS_MAP = {
    "us bk cacs": "u s bank",
    "us bk cac": "u s bank",
    "us bk cas": "u s bank",
    "us bk cc": "u s bank",
    "us bank cacs": "u s bank",
    "u.s. bank": "u s bank",
    "us bank": "u s bank",
}
_DATE_FORMATS = (
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%Y.%m.%d",
    "%m/%d/%Y",
    "%m-%d-%Y",
    "%m.%d.%Y",
    "%d.%m.%Y",
    "%d-%m-%Y",
    "%d/%m/%Y",
    "%m/%d/%y",
    "%m-%d-%y",
    "%Y%m%d",
)

_AMOUNT_FIELDS = {"past_due_amount", "high_balance", "credit_limit"}
_ZERO_AMOUNT_FIELDS = {"balance_owed", "past_due_amount"}
_AMOUNT_ZERO_EPSILON = 1e-9
_DATE_FIELDS_DET = {
    "last_verified",
    "date_of_last_activity",
    "date_reported",
    "date_opened",
    "closed_date",
}
_FIELD_SEQUENCE = (
    "balance_owed",
    "account_number",
    "last_payment",
    "past_due_amount",
    "high_balance",
    "creditor_type",
    "account_type",
    "payment_amount",
    "credit_limit",
    "last_verified",
    "date_of_last_activity",
    "date_reported",
    "date_opened",
    "closed_date",
)
_MID_FIELD_SET = {
    "last_payment",
    "past_due_amount",
    "high_balance",
    "creditor_type",
    "account_type",
    "payment_amount",
    "credit_limit",
}
_DATE_FIELDS_ORDER = (
    "last_verified",
    "date_of_last_activity",
    "date_reported",
    "date_opened",
    "closed_date",
)
_AMOUNT_CONFLICT_FIELDS = {
    "balance_owed",
    "payment_amount",
    "past_due_amount",
    "high_balance",
    "credit_limit",
}
_TYPE_FIELDS = {"creditor_type", "account_type"}


def _normalize_field_value(field: str, value: Any) -> Optional[Any]:
    """Normalize a merge field value according to deterministic rules."""

    if field == "balance_owed":
        return normalize_balance_owed(value)
    if field == "account_number":
        return normalize_account_number(value)
    if field == "payment_amount" or field in _AMOUNT_FIELDS:
        return normalize_amount_field(value)
    if field == "last_payment" or field in _DATE_FIELDS_DET:
        return to_date(value)
    if field in _TYPE_FIELDS:
        return normalize_type(value)
    return value


def _serialize_normalized_value(value: Any) -> Any:
    """Convert normalized values into JSON/log friendly primitives."""

    if isinstance(value, datetime):
        value = value.date()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return value


def _serialize_normalized_pair(a: Any, b: Any) -> Tuple[Any, Any]:
    return (_serialize_normalized_value(a), _serialize_normalized_value(b))


def _extract_account_number_string(
    bureaus: Mapping[str, Mapping[str, Any]],
    preferred_bureau: Optional[str] = None,
) -> str:
    """Return the best raw account-number string for a bureau mapping."""

    if not isinstance(bureaus, Mapping):
        return ""

    ordered_candidates: List[str] = []
    if preferred_bureau:
        ordered_candidates.append(str(preferred_bureau))
    ordered_candidates.extend(["transunion", "experian", "equifax"])

    seen: Set[str] = set()
    for bureau_key in ordered_candidates:
        if bureau_key in seen:
            continue
        seen.add(bureau_key)

        branch = bureaus.get(bureau_key)
        if not isinstance(branch, Mapping):
            continue

        for field_name in ("account_number_display", "account_number"):
            raw_value = branch.get(field_name)
            if is_missing(raw_value):
                continue
            return str(raw_value)

    return ""


def _normalize_account_display(branch: Mapping[str, Any] | None) -> acctnum.NormalizedAccountNumber:
    if not isinstance(branch, Mapping):
        return acctnum.normalize_display("")

    raw_value: Optional[str] = None
    for key in ("account_number_display", "account_number"):
        candidate = branch.get(key)
        if not is_missing(candidate):
            raw_value = str(candidate)
            break

    return acctnum.normalize_display(raw_value)


def _match_account_number_best_pair(
    A: Mapping[str, Mapping[str, Any]],
    B: Mapping[str, Mapping[str, Any]],
    cfg: MergeCfg,
) -> Tuple[bool, Dict[str, Any]]:
    bureaus = ("transunion", "experian", "equifax")
    bureau_positions = {name: idx for idx, name in enumerate(bureaus)}

    hard_enabled = bool(cfg.triggers.get("MERGE_AI_ON_HARD_ACCTNUM", True))

    normalized_a = {name: _normalize_account_display(A.get(name)) for name in bureaus}
    normalized_b = {name: _normalize_account_display(B.get(name)) for name in bureaus}

    best_match_aux: Dict[str, Any] | None = None
    best_match_rank = -1
    best_pair_rank: Tuple[int, int] | None = None
    best_any_aux: Dict[str, Any] | None = None
    best_any_rank = -1
    first_aux: Dict[str, Any] | None = None

    for left in bureaus:
        left_norm = normalized_a[left]
        if not left_norm.has_digits:
            continue
        for right in bureaus:
            right_norm = normalized_b[right]
            if not right_norm.has_digits:
                continue

            level = _sanitize_acct_level(acctnum.match_level(left_norm, right_norm))
            level_rank = _ACCOUNT_LEVEL_ORDER.get(level, 0)
            matched = hard_enabled and level == "exact_or_known_match"

            result_aux: Dict[str, Any] = {
                "best_pair": (left, right),
                "normalized_values": (
                    left_norm.digits,
                    right_norm.digits,
                ),
                "acctnum_level": level,
                "acctnum_digits_len_a": len(left_norm.digits),
                "acctnum_digits_len_b": len(right_norm.digits),
                "raw_values": {"a": left_norm.raw, "b": right_norm.raw},
            }

            if first_aux is None:
                first_aux = dict(result_aux)

            if level_rank > best_any_rank:
                best_any_rank = level_rank
                best_any_aux = dict(result_aux)

            if not matched:
                continue

            pick = False
            if best_match_aux is None:
                pick = True
            else:
                prev_level = _sanitize_acct_level(best_match_aux.get("acctnum_level"))
                prev_rank = _ACCOUNT_LEVEL_ORDER.get(prev_level, 0)
                if level_rank > prev_rank:
                    pick = True
                elif level_rank == prev_rank:
                    pair_rank = (bureau_positions[left], bureau_positions[right])
                    if best_pair_rank is None or pair_rank < best_pair_rank:
                        pick = True

            if pick:
                best_match_aux = dict(result_aux)
                best_pair_rank = (bureau_positions[left], bureau_positions[right])
                best_match_rank = level_rank

    if best_match_aux is not None:
        best_match_aux["matched"] = True
        return True, best_match_aux

    if best_any_aux is not None:
        best_any_aux["matched"] = False
        return False, best_any_aux

    if first_aux is not None:
        first_aux["matched"] = False
        return False, first_aux

    return False, {}


def soft_acct_suspicious(a_display: str, b_display: str) -> bool:
    matched, _ = acctnum.acctnum_visible_match(a_display, b_display)
    return matched


def _detect_soft_acct_match(
    left_bureaus: Mapping[str, Mapping[str, Any]],
    right_bureaus: Mapping[str, Mapping[str, Any]],
) -> bool:
    bureaus = ("transunion", "experian", "equifax")

    for left_key in bureaus:
        left_branch = left_bureaus.get(left_key)
        if not isinstance(left_branch, Mapping):
            continue
        left_candidates = []
        for field_name in ("account_number_display", "account_number"):
            candidate = left_branch.get(field_name)
            if is_missing(candidate):
                continue
            left_candidates.append(str(candidate))
        if not left_candidates:
            continue

        for right_key in bureaus:
            right_branch = right_bureaus.get(right_key)
            if not isinstance(right_branch, Mapping):
                continue
            right_candidates = []
            for field_name in ("account_number_display", "account_number"):
                candidate = right_branch.get(field_name)
                if is_missing(candidate):
                    continue
                right_candidates.append(str(candidate))
            if not right_candidates:
                continue

            for left_display in left_candidates:
                for right_display in right_candidates:
                    if soft_acct_suspicious(left_display, right_display):
                        return True

    return False


def _log_candidate_considered(
    sid: str,
    left: int,
    right: int,
    *,
    reason: str | None = None,
    record: Mapping[str, Any] | None = None,
    allow_flags: Mapping[str, Any] | None = None,
    total: Any | None = None,
    gate_level: str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> None:
    payload: Dict[str, Any] = {
        "sid": sid,
        "i": int(left),
        "j": int(right),
    }

    if record is not None:
        payload["allowed"] = bool(record.get("allowed"))
        payload["acctnum_level"] = _sanitize_acct_level(record.get("level"))
        payload["dates_all"] = bool(record.get("dates_all"))
        payload["score_gate"] = bool(record.get("score_gate"))
        payload["soft"] = bool(record.get("soft"))
        try:
            payload["mid_sum"] = int(record.get("mid_sum", 0) or 0)
        except (TypeError, ValueError):
            payload["mid_sum"] = 0
        try:
            payload["identity_sum"] = int(record.get("identity_sum", 0) or 0)
        except (TypeError, ValueError):
            payload["identity_sum"] = 0

        priority = record.get("priority")
        if isinstance(priority, Mapping):
            priority_payload: Dict[str, Any] = {}
            try:
                priority_payload["category"] = int(priority.get("category", 0) or 0)
            except (TypeError, ValueError):
                priority_payload["category"] = 0
            try:
                priority_payload["subscore"] = int(priority.get("subscore", 0) or 0)
            except (TypeError, ValueError):
                priority_payload["subscore"] = 0
            priority_payload["label"] = str(priority.get("label", ""))
            payload["priority"] = priority_payload

        if reason is None:
            reason = record.get("reason")

    if reason is not None:
        payload["reason"] = str(reason)

    if allow_flags is None and isinstance(record, Mapping):
        allow_flags = record.get("allow_flags")
    if isinstance(allow_flags, Mapping):
        payload["allow_flags"] = {
            "hard_acct": bool(allow_flags.get("hard_acct")),
            "dates": bool(allow_flags.get("dates")),
            "total": bool(allow_flags.get("total")),
        }

    if total is None and isinstance(record, Mapping):
        total = record.get("total", 0)
    try:
        payload["total"] = int(total or 0)
    except (TypeError, ValueError):
        payload["total"] = 0

    if gate_level is None and isinstance(record, Mapping):
        gate_level = record.get("level")
    if gate_level is not None:
        payload["acctnum_gate_level"] = _sanitize_acct_level(gate_level)

    if isinstance(extra, Mapping):
        for key, value in extra.items():
            payload[key] = value

    logger.info("CANDIDATE_CONSIDERED %s", json.dumps(payload, sort_keys=True))


def _log_candidate_skipped(
    sid: str,
    left: int,
    right: int,
    *,
    reason: str,
    record: Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> None:
    payload: Dict[str, Any] = {
        "sid": sid,
        "i": int(left),
        "j": int(right),
        "reason": str(reason),
    }

    if isinstance(record, Mapping):
        payload["allowed"] = bool(record.get("allowed"))
        payload["acctnum_level"] = _sanitize_acct_level(record.get("level"))
        try:
            payload["total"] = int(record.get("total", 0) or 0)
        except (TypeError, ValueError):
            payload["total"] = 0
        payload["dates_all"] = bool(record.get("dates_all"))
        payload["score_gate"] = bool(record.get("score_gate"))

        allow_flags = record.get("allow_flags")
        if isinstance(allow_flags, Mapping):
            payload["allow_flags"] = {
                "hard_acct": bool(allow_flags.get("hard_acct")),
                "dates": bool(allow_flags.get("dates")),
                "total": bool(allow_flags.get("total")),
            }

        priority = record.get("priority")
        if isinstance(priority, Mapping):
            priority_payload: Dict[str, Any] = {}
            try:
                priority_payload["category"] = int(priority.get("category", 0) or 0)
            except (TypeError, ValueError):
                priority_payload["category"] = 0
            try:
                priority_payload["subscore"] = int(priority.get("subscore", 0) or 0)
            except (TypeError, ValueError):
                priority_payload["subscore"] = 0
            priority_payload["label"] = str(priority.get("label", ""))
            payload["priority"] = priority_payload

    if isinstance(extra, Mapping):
        for key, value in extra.items():
            payload[key] = value

    logger.info("CANDIDATE_SKIPPED %s", json.dumps(payload, sort_keys=True))


def _is_zero_amount(value: Any) -> bool:
    try:
        return abs(float(value)) <= _AMOUNT_ZERO_EPSILON
    except (TypeError, ValueError):
        return False


def _both_amounts_positive(pair: Any) -> bool:
    if not isinstance(pair, (list, tuple)) or len(pair) != 2:
        return False
    try:
        return float(pair[0]) > 0 and float(pair[1]) > 0
    except (TypeError, ValueError):
        return False


def _match_field_values(
    field: str,
    norm_a: Any,
    norm_b: Any,
    raw_a: Any,
    raw_b: Any,
    cfg: MergeCfg,
) -> Tuple[bool, Dict[str, Any]]:
    """Apply the appropriate predicate for a normalized pair of values."""

    aux: Dict[str, Any] = {}

    if field == "balance_owed":
        matched = match_balance_owed(norm_a, norm_b)
        if matched and (_is_zero_amount(norm_a) or _is_zero_amount(norm_b)):
            return False, aux
        return matched, aux

    if field == "payment_amount":
        tol_abs = float(cfg.tolerances.get("AMOUNT_TOL_ABS", 0.0))
        tol_ratio = float(cfg.tolerances.get("AMOUNT_TOL_RATIO", 0.0))
        count_zero = int(cfg.tolerances.get("COUNT_ZERO_PAYMENT_MATCH", 0))
        matched = match_payment_amount(
            norm_a,
            norm_b,
            tol_abs=tol_abs,
            tol_ratio=tol_ratio,
            count_zero_payment_match=count_zero,
        )
        return matched, aux

    if field in _AMOUNT_FIELDS:
        tol_abs = float(cfg.tolerances.get("AMOUNT_TOL_ABS", 0.0))
        tol_ratio = float(cfg.tolerances.get("AMOUNT_TOL_RATIO", 0.0))
        matched = match_amount_field(norm_a, norm_b, tol_abs=tol_abs, tol_ratio=tol_ratio)
        if field in _ZERO_AMOUNT_FIELDS and (
            _is_zero_amount(norm_a) or _is_zero_amount(norm_b)
        ):
            return False, aux
        return matched, aux

    if field == "last_payment":
        day_tol = int(cfg.tolerances.get("LAST_PAYMENT_DAY_TOL", 0))
        matched = date_within(norm_a, norm_b, day_tol)
        return matched, aux

    if field in _DATE_FIELDS_DET:
        return date_equal(norm_a, norm_b), aux

    if field in _TYPE_FIELDS:
        matched = norm_a == norm_b and norm_a is not None and norm_b is not None
        return matched, aux

    raise KeyError(f"Unsupported merge field: {field}")


def match_field_best_of_9(
    field_name: str,
    A: Mapping[str, Mapping[str, Any]],
    B: Mapping[str, Mapping[str, Any]],
    cfg: MergeCfg,
) -> Tuple[bool, Dict[str, Any]]:
    """Check all cross-bureau pairs for a field and return best match metadata."""

    if not isinstance(A, Mapping):
        A = {}
    if not isinstance(B, Mapping):
        B = {}

    bureaus = ("transunion", "experian", "equifax")
    bureau_positions = {name: idx for idx, name in enumerate(bureaus)}
    field_key = str(field_name)

    if field_key == "account_number":
        return _match_account_number_best_pair(A, B, cfg)

    best_aux: Dict[str, Any] | None = None
    best_score = -1
    best_matched_aux: Dict[str, Any] | None = None
    best_matched_score = -1
    first_candidate_aux: Dict[str, Any] | None = None
    best_pair_rank: tuple[int, int] | None = None

    for left in bureaus:
        left_branch = A.get(left)
        if not isinstance(left_branch, Mapping):
            continue
        raw_left = left_branch.get(field_key)
        if field_key == "account_number" and is_missing(raw_left):
            raw_left = left_branch.get("account_number_display")
        if is_missing(raw_left):
            continue
        norm_left = _normalize_field_value(field_key, raw_left)
        if norm_left is None:
            continue

        for right in bureaus:
            right_branch = B.get(right)
            if not isinstance(right_branch, Mapping):
                continue
            raw_right = right_branch.get(field_key)
            if is_missing(raw_right):
                continue
            norm_right = _normalize_field_value(field_key, raw_right)
            if norm_right is None:
                continue

            matched, aux = _match_field_values(
                field_key, norm_left, norm_right, raw_left, raw_right, cfg
            )

            result_aux = {
                "best_pair": (left, right),
                "normalized_values": _serialize_normalized_pair(norm_left, norm_right),
            }
            result_aux.update(aux)

            if first_candidate_aux is None:
                first_candidate_aux = dict(result_aux)

            if field_key == "account_number":
                level = _sanitize_acct_level(aux.get("acctnum_level"))
                level_score = _ACCOUNT_LEVEL_ORDER.get(level, 0)
                if level_score > best_score:
                    best_score = level_score
                    best_aux = dict(result_aux)
                if matched and level_score >= best_matched_score:
                    best_matched_score = level_score
                    best_matched_aux = dict(result_aux)
                # Continue searching for a better match even after finding one.
                continue

            if matched:
                pair_rank = (bureau_positions[left], bureau_positions[right])
                if best_matched_aux is None or (
                    best_pair_rank is None or pair_rank < best_pair_rank
                ):
                    best_matched_aux = dict(result_aux)
                    best_pair_rank = pair_rank

    if best_matched_aux is not None:
        return True, best_matched_aux
    if first_candidate_aux is not None:
        return False, first_candidate_aux

    return False, {}


def _collect_normalized_field_values(
    bureaus: Mapping[str, Mapping[str, Any]], field: str
) -> List[Any]:
    values: List[Any] = []
    if not isinstance(bureaus, Mapping):
        return values

    for bureau_key in ("transunion", "experian", "equifax"):
        branch = bureaus.get(bureau_key)
        if not isinstance(branch, Mapping):
            continue
        raw_value = branch.get(field)
        if is_missing(raw_value):
            continue
        norm_value = _normalize_field_value(field, raw_value)
        if norm_value is None:
            continue
        values.append(norm_value)
    return values


def _detect_amount_conflicts(
    A: Mapping[str, Mapping[str, Any]],
    B: Mapping[str, Mapping[str, Any]],
    cfg: MergeCfg,
) -> List[str]:
    conflicts: List[str] = []
    tol_abs = float(cfg.tolerances.get("AMOUNT_TOL_ABS", 0.0))
    tol_ratio = float(cfg.tolerances.get("AMOUNT_TOL_RATIO", 0.0))

    for field in _AMOUNT_CONFLICT_FIELDS:
        values_a = _collect_normalized_field_values(A, field)
        values_b = _collect_normalized_field_values(B, field)
        if not values_a or not values_b:
            continue

        conflict = True
        if field == "balance_owed":
            for left in values_a:
                for right in values_b:
                    if match_balance_owed(left, right):
                        conflict = False
                        break
                if not conflict:
                    break
        else:
            for left in values_a:
                for right in values_b:
                    if match_amount_field(left, right, tol_abs=tol_abs, tol_ratio=tol_ratio):
                        conflict = False
                        break
                if not conflict:
                    break

        if conflict:
            conflicts.append(f"amount_conflict:{field}")

    return conflicts


def score_pair_0_100(
    A_bureaus: Mapping[str, Mapping[str, Any]],
    B_bureaus: Mapping[str, Mapping[str, Any]],
    cfg: MergeCfg,
) -> Dict[str, Any]:
    if not isinstance(A_bureaus, Mapping):
        A_data: Mapping[str, Mapping[str, Any]] = {}
    else:
        A_data = A_bureaus
    if not isinstance(B_bureaus, Mapping):
        B_data: Mapping[str, Mapping[str, Any]] = {}
    else:
        B_data = B_bureaus

    total = 0
    mid_sum = 0
    identity_sum = 0
    parts: Dict[str, int] = {}
    aux: Dict[str, Dict[str, Any]] = {}
    field_matches: Dict[str, bool] = {}
    date_matches: Dict[str, bool] = {field: False for field in _DATE_FIELDS_ORDER}
    trigger_events: List[Dict[str, Any]] = []

    acct_match_raw, acct_aux_raw = match_field_best_of_9(
        "account_number", A_data, B_data, cfg
    )
    acct_aux: Dict[str, Any] = dict(acct_aux_raw) if isinstance(acct_aux_raw, Mapping) else {}
    best_pair = acct_aux.get("best_pair") if isinstance(acct_aux, Mapping) else None
    left_pref: Optional[str]
    right_pref: Optional[str]
    if isinstance(best_pair, (list, tuple)) and len(best_pair) == 2:
        left_pref = str(best_pair[0])
        right_pref = str(best_pair[1])
    else:
        left_pref = None
        right_pref = None

    a_account_str = _extract_account_number_string(A_data, left_pref)
    b_account_str = _extract_account_number_string(B_data, right_pref)

    acct_level, acct_debug = acctnum_match_level(a_account_str, b_account_str)
    acct_points = int(_ACCOUNT_NUMBER_WEIGHTS.get(acct_level, 0) or 0)
    acct_matched = acct_level == "exact_or_known_match"

    field_matches["account_number"] = acct_matched
    parts["account_number"] = acct_points
    identity_sum += acct_points
    total += acct_points

    acct_aux.update(
        {
            "acctnum_level": acct_level,
            "matched": acct_matched,
            "acctnum_debug": acct_debug,
            "raw_values": {"a": a_account_str, "b": b_account_str},
            "threshold_matched": bool(acct_match_raw),
        }
    )
    aux["account_number"] = acct_aux

    for field in _FIELD_SEQUENCE:
        if field == "account_number":
            continue
        matched, match_aux = match_field_best_of_9(field, A_data, B_data, cfg)
        base_points = int(cfg.points.get(field, 0))
        awarded_points = 0
        if matched:
            awarded_points = base_points
            total += awarded_points
            if field in _MID_FIELD_SET:
                mid_sum += awarded_points
            if field in _IDENTITY_FIELD_SET:
                identity_sum += awarded_points
        parts[field] = awarded_points

        per_field_aux: Dict[str, Any] = dict(match_aux)
        per_field_aux["matched"] = matched
        aux[field] = per_field_aux
        field_matches[field] = matched

        if field in date_matches:
            date_matches[field] = matched

    dates_all = bool(date_matches) and all(date_matches.values())

    triggers: List[str] = []
    strong_triggered = False
    mid_triggered = False
    dates_triggered = False
    total_triggered = False

    balance_aux = aux.get("balance_owed", {})
    balance_pair = balance_aux.get("normalized_values") if isinstance(balance_aux, Mapping) else None
    if (
        cfg.triggers.get("MERGE_AI_ON_BALOWED_EXACT", True)
        and field_matches.get("balance_owed")
        and _both_amounts_positive(balance_pair)
    ):
        triggers.append("strong:balance_owed")
        strong_triggered = True
        trigger_events.append(
            {
                "kind": "strong",
                "details": {
                    "field": "balance_owed",
                    "points": int(cfg.points.get("balance_owed", 0)),
                },
            }
        )

    acctnum_aux = aux.get("account_number", {})
    acct_level = _sanitize_acct_level(acctnum_aux.get("acctnum_level"))
    if field_matches.get("account_number") and acct_level == "exact_or_known_match":
        triggers.append("strong:account_number")
        strong_triggered = True
        trigger_events.append(
            {
                "kind": "strong",
                "details": {
                    "field": "account_number",
                    "level": acct_level,
                },
            }
        )

    mid_threshold = int(cfg.triggers.get("MERGE_AI_ON_MID_K", 0))
    if mid_sum >= mid_threshold and mid_threshold > 0:
        triggers.append("mid")
        mid_triggered = True
        trigger_events.append(
            {
                "kind": "mid",
                "details": {
                    "mid_sum": int(mid_sum),
                    "threshold": int(mid_threshold),
                },
            }
        )

    if cfg.triggers.get("MERGE_AI_ON_ALL_DATES", False) and dates_all:
        triggers.append("dates")
        dates_triggered = True
        trigger_events.append(
            {
                "kind": "dates",
                "details": {
                    "matched_fields": [
                        field
                        for field, matched in date_matches.items()
                        if matched
                    ],
                    "required_all": True,
                },
            }
        )

    ai_threshold = int(cfg.thresholds.get("AI_THRESHOLD", 0))
    if total >= ai_threshold and ai_threshold > 0:
        triggers.append("total")
        total_triggered = True
        trigger_events.append(
            {
                "kind": "total",
                "details": {
                    "total": int(total),
                    "threshold": int(ai_threshold),
                },
            }
        )

    conflicts: List[str] = []
    for conflict in _detect_amount_conflicts(A_data, B_data, cfg):
        if conflict not in conflicts:
            conflicts.append(conflict)

    auto_threshold = int(cfg.thresholds.get("AUTO_MERGE_THRESHOLD", 0))
    has_hard_conflict = bool(conflicts)

    decision = "different"
    if total >= auto_threshold and auto_threshold > 0 and not has_hard_conflict:
        decision = "auto"
    else:
        ai_triggered = (
            strong_triggered
            or mid_triggered
            or dates_triggered
            or total_triggered
        )
        if ai_triggered:
            decision = "ai"

    result = {
        "total": int(total),
        "parts": parts,
        "mid_sum": int(mid_sum),
        "identity_sum": int(identity_sum),
        "identity_score": int(identity_sum),
        "dates_all": dates_all,
        "aux": aux,
        "triggers": triggers,
        "conflicts": conflicts,
        "decision": decision,
        "trigger_events": trigger_events,
    }

    return result


def _strong_priority(triggers: Iterable[str]) -> int:
    """Return a numeric priority for strong triggers."""

    trigger_set = set(triggers or [])
    if "strong:balance_owed" in trigger_set:
        return 2
    if "strong:account_number" in trigger_set:
        return 1
    return 0


def score_all_pairs_0_100(
    sid: str,
    idx_list: Iterable[int],
    runs_root: Path = Path("runs"),
) -> Dict[int, Dict[int, Dict[str, Any]]]:
    """Score all unordered account pairs for a case run."""

    cfg = get_merge_cfg()
    ai_threshold = int(cfg.thresholds.get("AI_THRESHOLD", 0) or 0)
    requested_raw = list(idx_list) if idx_list is not None else []
    requested_indices: List[int] = []
    for raw_idx in requested_raw:
        if isinstance(raw_idx, bool):
            continue
        try:
            idx_val = int(raw_idx)
        except (TypeError, ValueError):
            logger.warning("MERGE_V2_SCORE sid=<%s> invalid_index=%r", sid, raw_idx)
            continue
        requested_indices.append(idx_val)

    requested_set = set(requested_indices)

    accounts_root = runs_root / sid / "cases" / "accounts"
    discovered_indices: List[int] = []
    if accounts_root.exists():
        for entry in accounts_root.iterdir():
            if not entry.is_dir():
                continue
            try:
                idx_val = int(entry.name)
            except (TypeError, ValueError):
                logger.debug(
                    "MERGE_V2_SCORE sid=<%s> skip_account_dir=%r", sid, entry.name
                )
                continue
            discovered_indices.append(idx_val)
    else:
        logger.warning(
            "MERGE_V2_SCORE sid=<%s> accounts_dir_missing path=%s",
            sid,
            accounts_root,
        )

    if requested_raw:
        if requested_set:
            indices = sorted(idx for idx in set(discovered_indices) if idx in requested_set)
            missing = requested_set - set(indices)
            if missing:
                logger.debug(
                    "MERGE_V2_SCORE sid=<%s> missing_requested_indices=%s",
                    sid,
                    sorted(missing),
                )
        else:
            indices = []
    else:
        indices = sorted(set(discovered_indices))

    total_accounts = len(indices)
    expected_pairs = total_accounts * (total_accounts - 1) // 2

    overview_log = {
        "sid": sid,
        "indices": indices,
        "total_accounts": total_accounts,
        "expected_pairs": expected_pairs,
    }
    logger.debug("MERGE_PAIR_OVERVIEW %s", json.dumps(overview_log, sort_keys=True))

    bureaus_by_idx: Dict[int, Dict[str, Dict[str, Any]]] = {}
    for idx in indices:
        try:
            bureaus = load_bureaus(sid, idx, runs_root=runs_root)
        except FileNotFoundError:
            logger.warning(
                "MERGE_V2_SCORE sid=<%s> idx=<%s> bureaus_missing", sid, idx
            )
            bureaus = {}
        except Exception:
            logger.exception(
                "MERGE_V2_SCORE sid=<%s> idx=<%s> bureaus_load_failed", sid, idx
            )
            bureaus = {}
        bureaus_by_idx[idx] = bureaus

    scores: Dict[int, Dict[int, Dict[str, Any]]] = {idx: {} for idx in indices}

    logger.info("CANDIDATE_LOOP_START sid=%s total_accounts=%s", sid, total_accounts)

    pair_counter = 0
    candidate_records: List[Dict[str, Any]] = []
    for left_pos in range(total_accounts - 1):
        left = indices[left_pos]
        for right_pos in range(left_pos + 1, total_accounts):
            right = indices[right_pos]
            pair_counter += 1
            step_log = {
                "sid": sid,
                "i": left,
                "j": right,
                "pair_index": pair_counter,
                "expected_pairs": expected_pairs,
            }
            logger.debug("MERGE_PAIR_STEP %s", json.dumps(step_log, sort_keys=True))

            left_bureaus = bureaus_by_idx.get(left, {})
            right_bureaus = bureaus_by_idx.get(right, {})
            result = score_pair_0_100(
                left_bureaus,
                right_bureaus,
                cfg,
            )

            total_score = int(result.get("total", 0) or 0)
            sanitized_parts = _sanitize_parts(result.get("parts"))
            aux_payload = _build_aux_payload(result.get("aux", {}))

            acct_level = _sanitize_acct_level(aux_payload.get("acctnum_level"))
            raw_aux = result.get("aux") if isinstance(result.get("aux"), Mapping) else {}
            acct_aux = raw_aux.get("account_number") if isinstance(raw_aux, Mapping) else None
            if not isinstance(acct_aux, Mapping):
                acct_aux = {}
            raw_values: Dict[str, Any] = {}
            if isinstance(acct_aux, Mapping):
                raw_values_candidate = acct_aux.get("raw_values")
                if isinstance(raw_values_candidate, Mapping):
                    raw_values = raw_values_candidate
            a_acct_str = str(raw_values.get("a") or "")
            b_acct_str = str(raw_values.get("b") or "")
            if not a_acct_str:
                a_acct_str = _extract_account_number_string(left_bureaus)
            if not b_acct_str:
                b_acct_str = _extract_account_number_string(right_bureaus)
            gate_level, gate_detail = acctnum_match_level(a_acct_str, b_acct_str)
            level_candidate = (
                acct_aux.get("acctnum_level")
                or acct_level
                or gate_level
                or "none"
            )
            level_value = _sanitize_acct_level(level_candidate)
            debug_payload = {}
            if isinstance(acct_aux, Mapping):
                debug_candidate = acct_aux.get("acctnum_debug")
                if isinstance(debug_candidate, Mapping):
                    debug_payload = debug_candidate
            short_debug = ""
            long_debug = ""
            why_debug = ""
            if isinstance(debug_payload, Mapping):
                short_debug = str(debug_payload.get("short", ""))
                long_debug = str(debug_payload.get("long", ""))
                why_debug = str(debug_payload.get("why", ""))
            if not short_debug and isinstance(gate_detail, Mapping):
                short_debug = str(gate_detail.get("short", ""))
            if not long_debug and isinstance(gate_detail, Mapping):
                long_debug = str(gate_detail.get("long", ""))
            if not why_debug and isinstance(gate_detail, Mapping):
                why_debug = str(gate_detail.get("why", ""))
            logger.info(
                "MERGE_V2_ACCTNUM_MATCH sid=%s i=%s j=%s level=%s short=%s long=%s why=%s",
                sid,
                left,
                right,
                level_value,
                short_debug,
                long_debug,
                why_debug,
            )

            score_log = {
                "sid": sid,
                "i": left,
                "j": right,
                "total": total_score,
                "parts": sanitized_parts,
                "acctnum_level": acct_level,
                "matched_pairs": aux_payload.get("by_field_pairs", {}),
                "matched_fields": aux_payload.get("matched_fields", {}),
            }
            score_log["decision"] = str(result.get("decision", "different"))
            logger.info("MERGE_V2_SCORE %s", json.dumps(score_log, sort_keys=True))

            for event in result.get("trigger_events", []) or []:
                if not isinstance(event, Mapping):
                    continue
                kind = event.get("kind")
                trigger_log = {
                    "sid": sid,
                    "i": left,
                    "j": right,
                    "kind": kind,
                    "details": event.get("details", {}),
                }
                logger.info(
                    "MERGE_V2_TRIGGER %s",
                    json.dumps(trigger_log, sort_keys=True),
                )

            decision_log = {
                "sid": sid,
                "i": left,
                "j": right,
                "decision": str(result.get("decision", "different")),
                "total": total_score,
                "triggers": list(result.get("triggers", [])),
                "conflicts": list(result.get("conflicts", [])),
            }
            logger.info(
                "MERGE_V2_DECISION %s", json.dumps(decision_log, sort_keys=True)
            )

            hard_acct = gate_level == "exact_or_known_match"
            dates_all_equal = bool(result.get("dates_all"))
            allow_by_dates = dates_all_equal
            allow_by_total = ai_threshold > 0 and total_score >= ai_threshold

            mid_sum = int(result.get("mid_sum", 0) or 0)
            identity_sum = int(result.get("identity_sum", 0) or 0)
            soft_match = False
            if not hard_acct:
                soft_match = _detect_soft_acct_match(left_bureaus, right_bureaus)

            priority_category, priority_subscore, priority_label = _priority_category(
                level_value, allow_by_dates, allow_by_total
            )
            if soft_match and priority_label == "default":
                priority_label = "soft_acctnum"

            allow_flags = {
                "hard_acct": hard_acct,
                "dates": allow_by_dates,
                "total": allow_by_total,
            }

            def add_candidate(
                i: int,
                j: int,
                *,
                reason: str,
                allowed: bool = True,
            ) -> Dict[str, Any]:
                record = {
                    "left": i,
                    "right": j,
                    "result": deepcopy(result),
                    "allowed": allowed,
                    "allow_flags": dict(allow_flags),
                    "level": level_value,
                    "dates_all": dates_all_equal,
                    "score_gate": allow_by_total,
                    "total": total_score,
                    "mid_sum": mid_sum,
                    "identity_sum": identity_sum,
                    "soft": soft_match,
                    "reason": reason,
                    "priority": {
                        "category": priority_category,
                        "subscore": priority_subscore,
                        "label": priority_label,
                    },
                }
                candidate_records.append(record)
                _log_candidate_considered(
                    sid,
                    i,
                    j,
                    reason=reason,
                    record=record,
                    allow_flags=dict(allow_flags),
                    total=total_score,
                    gate_level=gate_level,
                )
                return record

            if hard_acct:
                add_candidate(left, right, reason="hard_acctnum")
                continue

            if allow_by_total:
                add_candidate(left, right, reason="score_gate")
            elif dates_all_equal:
                add_candidate(left, right, reason="dates_all_equal")
            else:
                record = add_candidate(
                    left,
                    right,
                    reason="below_threshold_no_acctnum",
                    allowed=False,
                )
                _log_candidate_skipped(
                    sid,
                    left,
                    right,
                    reason="below_threshold_no_acctnum",
                    record=record,
                )

    allowed_records = [record for record in candidate_records if record.get("allowed")]

    global_limit, per_account_limit = _read_candidate_limits()
    per_account_counts: Dict[int, int] = defaultdict(int)

    def _sort_key(record: Mapping[str, Any]) -> Tuple[int, ...]:
        priority = record.get("priority", {}) if isinstance(record.get("priority"), Mapping) else {}
        level_value = str(record.get("level", "none") or "none").lower()
        level_rank = _ACCOUNT_LEVEL_PRIORITY.get(level_value, 0)

        identity_raw = record.get("identity_score", record.get("identity_sum", 0))
        try:
            identity_val = int(identity_raw or 0)
        except (TypeError, ValueError):
            identity_val = 0

        mid_raw = record.get("mid_sum", 0)
        try:
            mid_val = int(mid_raw or 0)
        except (TypeError, ValueError):
            mid_val = 0

        total_raw = record.get("total", 0)
        try:
            total_val = int(total_raw or 0)
        except (TypeError, ValueError):
            total_val = 0

        dates_flag = 1 if record.get("dates_all") else 0
        score_gate_flag = 1 if record.get("score_gate") else 0

        category = int(priority.get("category", 3))
        subscore = int(priority.get("subscore", 0))

        left_idx = int(record.get("left", 0) or 0)
        right_idx = int(record.get("right", 0) or 0)

        return (
            -level_rank,
            -mid_val,
            -identity_val,
            -dates_flag,
            score_gate_flag,
            -total_val,
            category,
            -subscore,
            left_idx,
            right_idx,
        )

    sorted_records = sorted(allowed_records, key=_sort_key)

    hard_records: List[Dict[str, Any]] = []
    nonhard_records: List[Dict[str, Any]] = []
    for record in sorted_records:
        allow_flags = record.get("allow_flags", {})
        is_hard = False
        if isinstance(allow_flags, Mapping):
            is_hard = bool(allow_flags.get("hard_acct"))
        if is_hard:
            hard_records.append(record)
        else:
            nonhard_records.append(record)

    sorted_hard_records = sorted(hard_records, key=_sort_key)
    sorted_nonhard_records = sorted(nonhard_records, key=_sort_key)

    selected_records: List[Dict[str, Any]] = []
    dropped_records: List[Tuple[Dict[str, Any], str]] = []

    selected_records.extend(sorted_hard_records)

    remaining_global: Optional[int]
    if global_limit is None:
        remaining_global = None
    else:
        remaining_global = max(global_limit - len(sorted_hard_records), 0)

    selected_nonhard: List[Dict[str, Any]] = []

    for record in sorted_nonhard_records:
        left = int(record.get("left"))
        right = int(record.get("right"))

        if per_account_limit and (
            per_account_counts[left] >= per_account_limit
            or per_account_counts[right] >= per_account_limit
        ):
            dropped_records.append((record, "per_account_limit"))
            _log_candidate_skipped(
                sid,
                left,
                right,
                reason="cap_nonhard_per_account",
                record=record,
                extra={
                    "per_account_limit": int(per_account_limit or 0),
                    "per_account_count_left": int(per_account_counts[left]),
                    "per_account_count_right": int(per_account_counts[right]),
                },
            )
            continue

        if remaining_global is not None and len(selected_nonhard) >= remaining_global:
            dropped_records.append((record, "global_limit"))
            _log_candidate_skipped(
                sid,
                left,
                right,
                reason="cap_nonhard_global",
                record=record,
                extra={
                    "global_limit": int(global_limit or 0),
                    "selected_nonhard": len(selected_nonhard),
                },
            )
            continue

        selected_nonhard.append(record)
        per_account_counts[left] += 1
        per_account_counts[right] += 1

    selected_records.extend(selected_nonhard)

    for record in selected_records:
        left = int(record.get("left"))
        right = int(record.get("right"))
        result = record.get("result") if isinstance(record.get("result"), Mapping) else {}
        allow_flags = record.get("allow_flags") if isinstance(record.get("allow_flags"), Mapping) else {}
        hard_flag = bool(allow_flags.get("hard_acct"))
        total_flag = bool(allow_flags.get("total"))
        dates_flag = bool(allow_flags.get("dates"))
        logger.info(
            (
                "CANDIDATE_SELECTED sid=%s i=%s j=%s hard=%s total=%s dates=%s "
                "reason=%s score=%s"
            ),
            sid,
            left,
            right,
            hard_flag,
            total_flag,
            dates_flag,
            record.get("reason"),
            record.get("total"),
        )
        scores[left][right] = deepcopy(result)
        scores[right][left] = deepcopy(result)

    built_pairs = len(selected_records)

    if global_limit or per_account_limit:
        limit_log = {
            "sid": sid,
            "limit": global_limit or 0,
            "per_account": per_account_limit or 0,
            "considered": len(candidate_records),
            "eligible": len(allowed_records),
            "kept": built_pairs,
            "dropped": len(dropped_records),
        }
        logger.info("CANDIDATE_LIMIT_SUMMARY %s", json.dumps(limit_log, sort_keys=True))

        for record, reason in dropped_records[:10]:
            priority = record.get("priority", {})
            level_value = str(record.get("level", "none") or "none")

            identity_raw = record.get("identity_score", record.get("identity_sum", 0))
            try:
                identity_val = int(identity_raw or 0)
            except (TypeError, ValueError):
                identity_val = 0

            mid_raw = record.get("mid_sum", 0)
            try:
                mid_val = int(mid_raw or 0)
            except (TypeError, ValueError):
                mid_val = 0

            drop_log = {
                "sid": sid,
                "i": record.get("left"),
                "j": record.get("right"),
                "reason": reason,
                "priority": priority.get("label"),
                "total": record.get("total", 0),
                "level": level_value,
                "identity": identity_val,
                "mid": mid_val,
                "dates_all": bool(record.get("dates_all")),
                "score_gate": bool(record.get("score_gate")),
                "soft": bool(record.get("soft")),
            }
            logger.info("CANDIDATE_LIMIT_DROP %s", json.dumps(drop_log, sort_keys=True))

    summary_log = {
        "sid": sid,
        "total_accounts": total_accounts,
        "expected_pairs": expected_pairs,
        "pairs_scored": pair_counter,
        "pairs_allowed": len(allowed_records),
        "pairs_built": built_pairs,
    }
    logger.debug("MERGE_PAIR_SUMMARY %s", json.dumps(summary_log, sort_keys=True))

    logger.info("CANDIDATE_LOOP_END sid=%s built_pairs=%s", sid, built_pairs)

    return scores


def choose_best_partner(
    scores_by_idx: Mapping[int, Mapping[int, Mapping[str, Any]]]
) -> Dict[int, Dict[str, Any]]:
    """Select the best partner for each account using deterministic tie-breakers."""

    best_map: Dict[int, Dict[str, Any]] = {}
    for idx in sorted(scores_by_idx.keys()):
        partner_map = scores_by_idx.get(idx) or {}
        best_partner: Optional[int] = None
        best_priority = -1
        best_score = -1
        tiebreaker_reason = "none"
        best_result: Optional[Dict[str, Any]] = None

        for partner_idx in sorted(partner_map.keys()):
            if partner_idx == idx:
                continue
            result = partner_map.get(partner_idx)
            if not isinstance(result, Mapping):
                continue
            triggers = result.get("triggers") or []
            strong_rank = _strong_priority(triggers)
            total_score = int(result.get("total", 0) or 0)

            choose = False
            reason = tiebreaker_reason
            if best_partner is None:
                choose = True
                if strong_rank > 0:
                    reason = "strong"
                elif total_score > 0:
                    reason = "score"
                else:
                    reason = "index"
            elif strong_rank > best_priority:
                choose = True
                reason = "strong"
            elif strong_rank == best_priority:
                if total_score > best_score:
                    choose = True
                    reason = "score"
                elif total_score == best_score and partner_idx < best_partner:
                    choose = True
                    reason = "index"

            if not choose:
                continue

            best_partner = partner_idx
            best_priority = strong_rank
            best_score = total_score
            tiebreaker_reason = reason
            best_result = deepcopy(result)

        best_map[idx] = {
            "partner_index": best_partner,
            "result": best_result,
            "tiebreaker": tiebreaker_reason,
            "strong_rank": best_priority,
            "score_total": best_score if best_score >= 0 else 0,
        }

    return best_map


def _build_aux_payload(aux: Mapping[str, Any]) -> Dict[str, Any]:
    acct_level = _sanitize_acct_level(None)
    by_field_pairs: Dict[str, List[str]] = {}
    matched_fields: Dict[str, bool] = {"account_number": False}
    account_number_matched = False
    acct_digits_len_a: Optional[int] = None
    acct_digits_len_b: Optional[int] = None

    if isinstance(aux, Mapping):
        acct_aux = aux.get("account_number")
        if isinstance(acct_aux, Mapping):
            level = acct_aux.get("acctnum_level")
            if level is not None:
                acct_level = _sanitize_acct_level(level)
            if "matched" in acct_aux:
                account_number_matched = bool(acct_aux.get("matched"))
            elif acct_level != "none":
                account_number_matched = True
            len_a = acct_aux.get("acctnum_digits_len_a")
            len_b = acct_aux.get("acctnum_digits_len_b")
            try:
                acct_digits_len_a = int(len_a) if len_a is not None else acct_digits_len_a
            except (TypeError, ValueError):
                acct_digits_len_a = acct_digits_len_a
            try:
                acct_digits_len_b = int(len_b) if len_b is not None else acct_digits_len_b
            except (TypeError, ValueError):
                acct_digits_len_b = acct_digits_len_b

        for field in _FIELD_SEQUENCE:
            field_aux = aux.get(field) if isinstance(aux, Mapping) else None
            if not isinstance(field_aux, Mapping):
                continue
            if field == "account_number":
                level_value = field_aux.get("acctnum_level")
                level = (
                    acct_level if level_value is None else _sanitize_acct_level(level_value)
                )
                if level != "none":
                    acct_level = level
                if "matched" in field_aux:
                    account_number_matched = bool(field_aux.get("matched"))
                elif level != "none":
                    account_number_matched = True
            elif "matched" in field_aux:
                matched_fields[field] = bool(field_aux.get("matched"))
            best_pair = field_aux.get("best_pair")
            if best_pair and isinstance(best_pair, (list, tuple)) and len(best_pair) == 2:
                by_field_pairs[field] = [str(best_pair[0]), str(best_pair[1])]

    matched_fields["account_number"] = account_number_matched
    if "account_number" not in by_field_pairs:
        by_field_pairs["account_number"] = []

    payload: Dict[str, Any] = {
        "acctnum_level": acct_level,
        "by_field_pairs": by_field_pairs,
        "matched_fields": matched_fields,
    }

    if acct_digits_len_a is not None:
        payload["acctnum_digits_len_a"] = acct_digits_len_a
    if acct_digits_len_b is not None:
        payload["acctnum_digits_len_b"] = acct_digits_len_b

    return payload


def _build_ai_highlights(result: Mapping[str, Any] | None) -> Dict[str, Any]:
    result_payload: Mapping[str, Any] = result if isinstance(result, Mapping) else {}

    total = _tag_safe_int(result_payload.get("total"))

    triggers_raw = result_payload.get("triggers", []) or []
    if isinstance(triggers_raw, Iterable) and not isinstance(triggers_raw, (str, bytes)):
        triggers = [str(item) for item in triggers_raw if item is not None]
    else:
        triggers = []

    conflicts_raw = result_payload.get("conflicts", []) or []
    if isinstance(conflicts_raw, Iterable) and not isinstance(conflicts_raw, (str, bytes)):
        conflicts = [str(item) for item in conflicts_raw if item is not None]
    else:
        conflicts = []

    parts = _sanitize_parts(result_payload.get("parts"))
    aux_payload = _build_aux_payload(result_payload.get("aux", {}))

    matched_fields_raw = aux_payload.get("matched_fields", {})
    if isinstance(matched_fields_raw, Mapping):
        matched_fields = {
            str(field): bool(flag) for field, flag in matched_fields_raw.items()
        }
    else:
        matched_fields = {}

    acctnum_level = _sanitize_acct_level(aux_payload.get("acctnum_level"))

    return {
        "total": total,
        "triggers": triggers,
        "parts": parts,
        "matched_fields": matched_fields,
        "conflicts": conflicts,
        "acctnum_level": acctnum_level,
    }


def _sanitize_parts(parts: Optional[Mapping[str, Any]]) -> Dict[str, int]:
    values: Dict[str, int] = {}
    for field in _FIELD_SEQUENCE:
        value = 0
        if isinstance(parts, Mapping):
            try:
                value = int(parts.get(field, 0) or 0)
            except (TypeError, ValueError):
                value = 0
        values[field] = value
    return values


def _build_pair_entry(partner_idx: int, result: Mapping[str, Any]) -> Dict[str, Any]:
    """Build a stable payload describing the merge relationship with a partner."""

    total = 0
    mid_value = 0
    dates_all = False
    triggers_raw: Iterable[Any] = []
    decision = "different"
    aux_payload: Mapping[str, Any] = {}
    parts_payload: Mapping[str, Any] = {}

    if isinstance(result, Mapping):
        try:
            total = int(result.get("total", 0) or 0)
        except (TypeError, ValueError):
            total = 0
        try:
            mid_value = int(result.get("mid_sum", 0) or 0)
        except (TypeError, ValueError):
            mid_value = 0
        dates_all = bool(result.get("dates_all"))
        decision = str(result.get("decision", "different"))
        aux_candidate = result.get("aux")
        if isinstance(aux_candidate, Mapping):
            aux_payload = aux_candidate
        parts_candidate = result.get("parts")
        if isinstance(parts_candidate, Mapping):
            parts_payload = parts_candidate
        triggers_candidate = result.get("triggers", []) or []
        if isinstance(triggers_candidate, Iterable):
            triggers_raw = triggers_candidate

    triggers = [str(trigger) for trigger in triggers_raw if trigger is not None]
    strong = any(trigger.startswith("strong:") for trigger in triggers)
    sanitized_parts = _sanitize_parts(parts_payload)
    aux_slim = _build_aux_payload(aux_payload)
    acct_level = _sanitize_acct_level(aux_slim.get("acctnum_level"))

    entry: Dict[str, Any] = {
        "with": int(partner_idx),
        "total": total,
        "decision": decision,
        "strong": strong,
        "mid": mid_value,
        "dates_all": dates_all,
        "acctnum_level": acct_level,
        "reasons": triggers,
        "parts": sanitized_parts,
    }

    return entry


def _build_best_match_entry(
    best_info: Optional[Mapping[str, Any]],
    pair_entries: Mapping[int, Mapping[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Construct the best-match payload leveraging computed pair entries."""

    if not isinstance(best_info, Mapping):
        return None

    partner = best_info.get("partner_index")
    if not isinstance(partner, int):
        return None

    entry = pair_entries.get(partner)
    total = 0
    decision = "different"

    acct_level = _sanitize_acct_level(None)

    if isinstance(entry, Mapping):
        try:
            total = int(entry.get("total", 0) or 0)
        except (TypeError, ValueError):
            total = 0
        decision = str(entry.get("decision", "different"))
        acct_level = _sanitize_acct_level(entry.get("acctnum_level", acct_level))
    else:
        result_payload = best_info.get("result")
        if isinstance(result_payload, Mapping):
            try:
                total = int(result_payload.get("total", 0) or 0)
            except (TypeError, ValueError):
                total = 0
            decision = str(result_payload.get("decision", "different"))
            aux_payload = _build_aux_payload(result_payload.get("aux", {}))
            acct_level = _sanitize_acct_level(aux_payload.get("acctnum_level", acct_level))
        try:
            total = int(best_info.get("score_total", total) or total)
        except (TypeError, ValueError):
            pass

    tiebreaker = str(best_info.get("tiebreaker", "none"))

    return {
        "with": int(partner),
        "total": total,
        "decision": decision,
        "tiebreaker": tiebreaker,
        "acctnum_level": acct_level,
    }


def _tag_safe_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _tag_normalize_str_list(values: Any) -> List[str]:
    if isinstance(values, Iterable) and not isinstance(values, (str, bytes)):
        return [str(item) for item in values if item is not None]
    return []


def _tag_normalize_merge_parts(parts: Any) -> Dict[str, int]:
    normalized: Dict[str, int] = {}
    if isinstance(parts, Mapping):
        for key in sorted(parts.keys(), key=str):
            normalized[str(key)] = _tag_safe_int(parts.get(key))
    return normalized


def _tag_normalize_merge_aux(aux: Any) -> Dict[str, Any]:
    acct_level = _sanitize_acct_level(None)
    by_field_pairs: Dict[str, List[str]] = {}
    matched_fields: Dict[str, bool] = {"account_number": False}
    account_number_matched = False
    acct_digits_len_a: Optional[int] = None
    acct_digits_len_b: Optional[int] = None

    if isinstance(aux, Mapping):
        acct_aux = aux.get("account_number")
        if isinstance(acct_aux, Mapping):
            level = acct_aux.get("acctnum_level")
            if level is not None:
                acct_level = _sanitize_acct_level(level)
            if "matched" in acct_aux:
                account_number_matched = bool(acct_aux.get("matched"))
            elif acct_level != "none":
                account_number_matched = True
            len_a = acct_aux.get("acctnum_digits_len_a")
            len_b = acct_aux.get("acctnum_digits_len_b")
            try:
                acct_digits_len_a = int(len_a) if len_a is not None else acct_digits_len_a
            except (TypeError, ValueError):
                acct_digits_len_a = acct_digits_len_a
            try:
                acct_digits_len_b = int(len_b) if len_b is not None else acct_digits_len_b
            except (TypeError, ValueError):
                acct_digits_len_b = acct_digits_len_b

        for field, field_aux in aux.items():
            if not isinstance(field_aux, Mapping):
                continue
            field_name = str(field)
            if field_name == "account_number":
                level_value = field_aux.get("acctnum_level")
                if level_value is not None:
                    level = _sanitize_acct_level(level_value)
                    if level != "none":
                        acct_level = level
                if "matched" in field_aux:
                    account_number_matched = bool(field_aux.get("matched"))
                elif account_number_matched:
                    account_number_matched = True
            elif "matched" in field_aux:
                matched_fields[field_name] = bool(field_aux.get("matched"))
            best_pair = field_aux.get("best_pair")
            if (
                isinstance(best_pair, (list, tuple))
                and len(best_pair) == 2
                and all(part is not None for part in best_pair)
            ):
                by_field_pairs[field_name] = [
                    str(best_pair[0]),
                    str(best_pair[1]),
                ]

    matched_fields["account_number"] = account_number_matched
    if "account_number" not in by_field_pairs:
        by_field_pairs["account_number"] = []

    payload: Dict[str, Any] = {
        "acctnum_level": acct_level,
        "by_field_pairs": by_field_pairs,
        "matched_fields": matched_fields,
    }

    if acct_digits_len_a is not None:
        payload["acctnum_digits_len_a"] = acct_digits_len_a
    if acct_digits_len_b is not None:
        payload["acctnum_digits_len_b"] = acct_digits_len_b

    return payload


def _normalize_merge_payload_for_tag(
    result: Mapping[str, Any] | None,
) -> Dict[str, Any]:
    payload = {
        "decision": "different",
        "total": 0,
        "mid": 0,
        "dates_all": False,
        "parts": {},
        "aux": {
            "acctnum_level": "none",
            "by_field_pairs": {},
            "matched_fields": {},
        },
        "reasons": [],
        "conflicts": [],
        "strong": False,
        "matched_pairs": {},
    }

    if isinstance(result, Mapping):
        payload["decision"] = str(result.get("decision", "different"))
        payload["total"] = _tag_safe_int(result.get("total"))
        payload["mid"] = _tag_safe_int(result.get("mid_sum"))
        payload["dates_all"] = bool(result.get("dates_all"))
        payload["parts"] = _tag_normalize_merge_parts(result.get("parts"))
        payload["aux"] = _tag_normalize_merge_aux(result.get("aux"))
        payload["reasons"] = _tag_normalize_str_list(result.get("triggers"))
        payload["conflicts"] = _tag_normalize_str_list(result.get("conflicts"))

    payload["strong"] = any(
        isinstance(reason, str) and reason.startswith("strong:")
        for reason in payload["reasons"]
    )

    aux_payload = payload.get("aux")
    acct_level = _sanitize_acct_level(None)
    if isinstance(aux_payload, Mapping):
        acct_level = _sanitize_acct_level(aux_payload.get("acctnum_level"))
        matched_pairs = aux_payload.get("by_field_pairs", {})
        if isinstance(matched_pairs, Mapping):
            payload["matched_pairs"] = {
                str(field): [str(pair[0]), str(pair[1])]
                for field, pair in matched_pairs.items()
                if isinstance(pair, (list, tuple))
                and len(pair) == 2
            }
        for key in ("acctnum_digits_len_a", "acctnum_digits_len_b"):
            value = aux_payload.get(key)
            if value is None:
                continue
            try:
                payload[key] = int(value)
            except (TypeError, ValueError):
                continue
    payload.setdefault("matched_pairs", {})
    payload["matched_pairs"].setdefault("account_number", [])
    payload["acctnum_level"] = acct_level

    return payload


def build_merge_pair_tag(partner_idx: int, result: Mapping[str, Any]) -> Dict[str, Any]:
    payload = _normalize_merge_payload_for_tag(result)
    payload.update(
        {
            "tag": "merge_pair",
            "kind": "merge_pair",
            "source": "merge_scorer",
            "with": int(partner_idx),
        }
    )
    return payload


def build_merge_best_tag(best_info: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(best_info, Mapping):
        return None

    partner = best_info.get("partner_index")
    result = best_info.get("result")
    if not isinstance(partner, int) or not isinstance(result, Mapping):
        return None

    payload = _normalize_merge_payload_for_tag(result)
    payload.update(
        {
            "tag": "merge_best",
            "kind": "merge_best",
            "source": "merge_scorer",
            "with": int(partner),
            "tiebreaker": str(best_info.get("tiebreaker", "none")),
            "strong_rank": _tag_safe_int(best_info.get("strong_rank")),
            "score_total": _tag_safe_int(
                best_info.get("score_total", payload["total"])
            ),
        }
    )
    return payload


def _build_score_entries(
    partner_scores: Mapping[int, Mapping[str, Any]],
    best_partner: Optional[int],
) -> List[Dict[str, Any]]:
    entries: Dict[int, Dict[str, Any]] = {}
    for partner_idx, result in partner_scores.items():
        if partner_idx == best_partner:
            # handle separately to ensure copy from best result later
            continue
        if not isinstance(result, Mapping):
            continue
        entry = {
            "account_index": partner_idx,
            "score": int(result.get("total", 0) or 0),
            "decision": str(result.get("decision", "different")),
            "triggers": list(result.get("triggers", [])),
            "conflicts": list(result.get("conflicts", [])),
        }
        entries[partner_idx] = entry

    sorted_entries = sorted(
        entries.values(), key=lambda item: (-item["score"], item["account_index"])
    )
    return sorted_entries


def _merge_tag_from_best(
    idx: int,
    partner_scores: Mapping[int, Mapping[str, Any]],
    best_info: Mapping[str, Any],
) -> Dict[str, Any]:
    best_partner = best_info.get("partner_index")
    best_result = best_info.get("result")
    tiebreaker = str(best_info.get("tiebreaker") or "none")

    if not isinstance(best_partner, int) or not isinstance(best_result, Mapping):
        parts = {field: 0 for field in _FIELD_SEQUENCE}
        merge_tag = {
            "group_id": f"g{idx}",
            "decision": "different",
            "score_total": 0,
            "score_to": _build_score_entries(partner_scores, None),
            "parts": parts,
            "aux": {"acctnum_level": "none", "by_field_pairs": {}, "matched_fields": {}},
            "reasons": [],
            "tiebreaker": "none",
        }
        merge_tag["acctnum_level"] = "none"
        merge_tag["matched_pairs"] = {"account_number": []}
        return merge_tag

    score_total = int(best_result.get("total", 0) or 0)
    decision = str(best_result.get("decision", "different"))
    triggers = list(best_result.get("triggers", []))
    conflicts = list(best_result.get("conflicts", []))
    parts = _sanitize_parts(best_result.get("parts"))
    aux_payload = _build_aux_payload(best_result.get("aux", {}))
    acct_level = _sanitize_acct_level(aux_payload.get("acctnum_level"))

    score_entries = _build_score_entries(partner_scores, best_partner)
    best_entry = {
        "account_index": best_partner,
        "score": score_total,
        "decision": decision,
        "triggers": triggers,
        "conflicts": conflicts,
    }
    score_to = [best_entry] + score_entries

    reasons = list(triggers)
    if conflicts:
        reasons.extend([f"conflict:{name}" for name in conflicts])

    merge_tag = {
        "group_id": f"g{idx}",
        "decision": decision,
        "score_total": score_total,
        "score_to": score_to,
        "parts": parts,
        "aux": aux_payload,
        "reasons": reasons,
        "tiebreaker": tiebreaker,
    }
    merge_tag["acctnum_level"] = acct_level
    matched_pairs_raw = aux_payload.get("by_field_pairs", {})
    pairs_payload: Dict[str, List[str]] = {}
    if isinstance(matched_pairs_raw, Mapping):
        for field, pair in matched_pairs_raw.items():
            if (
                isinstance(pair, (list, tuple))
                and len(pair) == 2
                and all(part is not None for part in pair)
            ):
                pairs_payload[str(field)] = [str(pair[0]), str(pair[1])]
    pairs_payload.setdefault("account_number", [])
    merge_tag["matched_pairs"] = pairs_payload

    for key in ("acctnum_digits_len_a", "acctnum_digits_len_b"):
        value = aux_payload.get(key)
        if value is None:
            continue
        try:
            merge_tag[key] = int(value)
        except (TypeError, ValueError):
            continue
    return merge_tag


def persist_merge_tags(
    sid: str,
    scores_by_idx: Mapping[int, Mapping[int, Mapping[str, Any]]],
    best_by_idx: Mapping[int, Mapping[str, Any]],
    runs_root: Path = Path("runs"),
) -> Dict[int, Dict[str, Any]]:
    """Persist merge tags for each account based on best-partner selection."""

    merge_tags: Dict[int, Dict[str, Any]] = {}
    all_indices = sorted(set(scores_by_idx.keys()) | set(best_by_idx.keys()))

    tags_root = runs_root / sid / "cases" / "accounts"
    tag_paths: Dict[int, Path] = {
        idx: tags_root / str(idx) / "tags.json" for idx in all_indices
    }

    merge_kinds = {"merge_pair", "merge_best"}
    for idx, path in tag_paths.items():
        existing_tags = read_tags(path)
        filtered = [tag for tag in existing_tags if tag.get("kind") not in merge_kinds]
        if filtered != existing_tags:
            write_tags_atomic(path, filtered)

    valid_decisions = {"ai", "auto"}
    processed_pairs: Set[Tuple[int, int]] = set()
    for left in sorted(scores_by_idx.keys()):
        partner_map = scores_by_idx.get(left) or {}
        for right in sorted(partner_map.keys()):
            if right == left or not isinstance(right, int):
                continue
            ordered = (min(left, right), max(left, right))
            if ordered in processed_pairs:
                continue
            processed_pairs.add(ordered)

            result = partner_map.get(right)
            if not isinstance(result, Mapping):
                continue

            left_tag = build_merge_pair_tag(right, result)
            if left_tag.get("decision") in valid_decisions:
                left_path = tag_paths.get(left)
                if left_path is not None:
                    upsert_tag(left_path, left_tag, unique_keys=("kind", "with"))

                right_tag = build_merge_pair_tag(left, result)
                right_path = tag_paths.get(right)
                if right_path is not None:
                    upsert_tag(right_path, right_tag, unique_keys=("kind", "with"))

                if left_tag.get("decision") == "ai":
                    highlights_from_pair = _build_ai_highlights(result)
                    pack_payload = build_ai_pack_for_pair(
                        sid,
                        runs_root,
                        left,
                        right,
                        highlights_from_pair,
                    )
                    try:
                        context_payload = pack_payload.get("context", {}) if isinstance(
                            pack_payload, Mapping
                        ) else {}
                        highlights_payload = pack_payload.get("highlights", {}) if isinstance(
                            pack_payload, Mapping
                        ) else {}

                        context_a = list(context_payload.get("a") or [])
                        context_b = list(context_payload.get("b") or [])

                        total_value = highlights_payload.get("total")
                        try:
                            total_value = int(total_value)
                        except (TypeError, ValueError):
                            total_value = None

                        triggers_raw = highlights_payload.get("triggers", [])
                        if isinstance(triggers_raw, Iterable) and not isinstance(
                            triggers_raw, (str, bytes)
                        ):
                            triggers_value = [str(item) for item in triggers_raw if item is not None]
                        else:
                            triggers_value = []

                        pack_log = {
                            "sid": sid,
                            "pair": {"a": left, "b": right},
                            "lines_a": len(context_a),
                            "lines_b": len(context_b),
                            "total": total_value,
                            "triggers": triggers_value,
                        }
                        logger.info("MERGE_V2_PACK %s", json.dumps(pack_log, sort_keys=True))
                    except Exception:  # pragma: no cover - defensive logging
                        logger.exception(
                            "MERGE_V2_PACK_FAILED sid=%s pair=(%s,%s)", sid, left, right
                        )

    for idx in all_indices:
        best_info = best_by_idx.get(idx, {})
        best_tag = build_merge_best_tag(best_info)
        if not best_tag or best_tag.get("decision") not in valid_decisions:
            continue
        path = tag_paths.get(idx)
        if path is not None:
            upsert_tag(path, best_tag, unique_keys=("kind",))

    for idx in all_indices:
        partner_scores = scores_by_idx.get(idx, {})
        best_info = best_by_idx.get(idx, {})
        merge_tag = _merge_tag_from_best(idx, partner_scores, best_info)
        merge_tags[idx] = merge_tag

    return merge_tags


def score_and_tag_best_partners(
    sid: str,
    idx_list: Iterable[int],
    runs_root: Path = Path("runs"),
) -> Dict[int, Dict[str, Any]]:
    """Convenience wrapper to score accounts, pick best partners, and persist tags."""

    scores = score_all_pairs_0_100(sid, idx_list, runs_root=runs_root)
    best = choose_best_partner(scores)
    return persist_merge_tags(sid, scores, best, runs_root=runs_root)


def to_amount(value: Any) -> Optional[float]:
    """Normalize free-form amount text to a float."""

    if is_missing(value):
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
    if not cleaned:
        return None

    match = _AMOUNT_RE.search(cleaned)
    if not match:
        return None

    number_text = match.group()
    try:
        number = float(number_text)
    except ValueError:
        return None

    if negative and number >= 0:
        number = -number

    return number


def amounts_match(a: Optional[float], b: Optional[float], tol_abs: float, tol_ratio: float) -> bool:
    """Return True when two normalized amounts match within tolerance."""

    if a is None or b is None:
        return False

    tol_abs = max(float(tol_abs), 0.0)
    tol_ratio = max(float(tol_ratio), 0.0)
    base = min(abs(a), abs(b))
    allowed = max(tol_abs, base * tol_ratio)
    return abs(a - b) <= allowed


def normalize_balance_owed(value: Any) -> Optional[float]:
    return to_amount(value)


def match_balance_owed(a: Optional[float], b: Optional[float]) -> bool:
    if a is None or b is None:
        return False
    return a == b


def normalize_amount_field(value: Any) -> Optional[float]:
    return to_amount(value)


def match_amount_field(
    a: Optional[float],
    b: Optional[float],
    *,
    tol_abs: float,
    tol_ratio: float,
) -> bool:
    return amounts_match(a, b, tol_abs, tol_ratio)


def match_payment_amount(
    a: Optional[float],
    b: Optional[float],
    *,
    tol_abs: float,
    tol_ratio: float,
    count_zero_payment_match: int,
) -> bool:
    if a is None or b is None:
        return False
    if not count_zero_payment_match and a == 0 and b == 0:
        return False
    return amounts_match(a, b, tol_abs, tol_ratio)


def digits_only(value: Any) -> Optional[str]:
    if is_missing(value):
        return None
    digits = re.sub(r"\D", "", str(value))
    return digits or None


def normalize_account_number(value: Any) -> Optional[str]:
    if is_missing(value):
        return None
    normalized = _normalize_acctnum_basic(str(value))
    return normalized


def account_number_level(a: Any, b: Any) -> str:
    if is_missing(a) or is_missing(b):
        return "none"

    level, _ = acctnum_match_level(str(a), str(b))
    return level


def account_numbers_match(
    a: Any, b: Any, min_level: str = "exact_or_known_match"
) -> Tuple[bool, str]:
    if is_missing(a) or is_missing(b):
        return False, "none"

    level, _ = acctnum_match_level(str(a), str(b))
    threshold = _ACCOUNT_LEVEL_ORDER.get(min_level, 0)
    match = level != "none" and _ACCOUNT_LEVEL_ORDER.get(level, 0) >= threshold
    return match, level


def to_date(value: Any) -> Optional[date]:
    if is_missing(value):
        return None
    if isinstance(value, date):
        return value
    if isinstance(value, datetime):
        return value.date()

    text = str(value).strip()
    if not text:
        return None

    if "T" in text:
        text = text.split("T", 1)[0]
    if " " in text and len(text.split()) > 1:
        text = text.split()[0]

    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue

    alt = re.sub(r"[.\\-]", "/", text)
    for fmt in ("%Y/%m/%d", "%m/%d/%Y", "%m/%d/%y", "%d/%m/%Y"):
        try:
            return datetime.strptime(alt, fmt).date()
        except ValueError:
            continue

    return None


def date_equal(a: Optional[date], b: Optional[date]) -> bool:
    if a is None or b is None:
        return False
    return a == b


def date_within(a: Optional[date], b: Optional[date], days: int) -> bool:
    if a is None or b is None:
        return False
    days = max(int(days), 0)
    return abs((a - b).days) <= days


def normalize_type(value: Any) -> Optional[str]:
    if is_missing(value):
        return None

    text = str(value).strip().lower()
    if not text:
        return None

    text = text.replace("\u2013", "-").replace("\u2014", "-")
    text = re.sub(r"[._]", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = text.strip("- ")
    if not text:
        return None

    alias_key = text.replace("-", " ")
    alias_key = re.sub(r"\s+", " ", alias_key)
    alias = _TYPE_ALIAS_MAP.get(alias_key)
    if alias:
        return alias

    normalized = alias_key
    normalized = normalized.replace("creditcard", "credit card")
    normalized = re.sub(r"\s+", " ", normalized).strip()

    if "credit card" in normalized:
        return "credit card"

    return normalized or None

