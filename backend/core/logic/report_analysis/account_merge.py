"""Tools for scoring and clustering problematic accounts prior to merging."""

from __future__ import annotations

import difflib
import json
import logging
import math
import os
import re
from copy import deepcopy
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from itertools import combinations
from typing import Any, Dict, Iterable, List, Mapping, Optional, Set, Tuple, Union

from backend.pipeline.runs import RunManifest


logger = logging.getLogger(__name__)


def is_missing(value: Any) -> bool:
    """Return True when a value represents an explicit missing sentinel."""

    return value in {None, "", "--"}


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
    "account_number": 28,
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
    "MERGE_AI_ON_ACCTNUM_LEVEL": "last4",
    "MERGE_AI_ON_MID_K": 26,
    "MERGE_AI_ON_ALL_DATES": 1,
}

_TOLERANCE_DEFAULTS: Dict[str, Union[int, float]] = {
    "AMOUNT_TOL_ABS": 50.0,
    "AMOUNT_TOL_RATIO": 0.01,
    "LAST_PAYMENT_DAY_TOL": 7,
    "COUNT_ZERO_PAYMENT_MATCH": 0,
}

_ACCTNUM_LEVEL_CHOICES: Set[str] = {"any", "last4", "exact"}


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
    triggers["MERGE_AI_ON_ACCTNUM_LEVEL"] = _read_env_choice(
        env_mapping,
        "MERGE_AI_ON_ACCTNUM_LEVEL",
        str(_TRIGGER_DEFAULTS["MERGE_AI_ON_ACCTNUM_LEVEL"]),
        _ACCTNUM_LEVEL_CHOICES,
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
_ACCOUNT_LEVEL_ORDER = {"none": 0, "any": 1, "last4": 2, "exact": 3}
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
_ACCOUNT_LAST4_KEYS = (
    "account_number_last4",
    "acct_last4",
    "account_last4",
    "last4",
    "account_number",
)
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

    if field == "account_number":
        min_level = str(cfg.triggers.get("MERGE_AI_ON_ACCTNUM_LEVEL", "last4")).lower()
        matched, level = account_numbers_match(raw_a, raw_b, min_level=min_level)
        aux["acctnum_level"] = level
        return matched, aux

    if field == "balance_owed":
        return match_balance_owed(norm_a, norm_b), aux

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
    field_key = str(field_name)

    for left in bureaus:
        left_branch = A.get(left)
        if not isinstance(left_branch, Mapping):
            continue
        raw_left = left_branch.get(field_key)
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
            if not matched:
                continue

            result_aux = {
                "best_pair": (left, right),
                "normalized_values": _serialize_normalized_pair(norm_left, norm_right),
            }
            result_aux.update(aux)
            return True, result_aux

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


def _collect_account_last4(bureaus: Mapping[str, Mapping[str, Any]]) -> Set[str]:
    last4_values: Set[str] = set()
    if not isinstance(bureaus, Mapping):
        return last4_values

    for bureau_key in ("transunion", "experian", "equifax"):
        branch = bureaus.get(bureau_key)
        if not isinstance(branch, Mapping):
            continue
        for key in _ACCOUNT_LAST4_KEYS:
            raw_value = branch.get(key)
            if is_missing(raw_value):
                continue
            digits = digits_only(raw_value)
            if not digits or len(digits) < 4:
                continue
            last4_values.add(digits[-4:])
    return last4_values


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


def score_pair(
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
    parts: Dict[str, int] = {}
    aux: Dict[str, Dict[str, Any]] = {}
    field_matches: Dict[str, bool] = {}
    date_matches: Dict[str, bool] = {field: False for field in _DATE_FIELDS_ORDER}

    for field in _FIELD_SEQUENCE:
        matched, match_aux = match_field_best_of_9(field, A_data, B_data, cfg)
        points = int(cfg.points.get(field, 0))
        if matched:
            total += points
            if field in _MID_FIELD_SET:
                mid_sum += points
        parts[field] = points if matched else 0

        per_field_aux: Dict[str, Any] = dict(match_aux)
        per_field_aux["matched"] = matched
        aux[field] = per_field_aux
        field_matches[field] = matched

        if field in date_matches:
            date_matches[field] = matched

    dates_all = bool(date_matches) and all(date_matches.values())

    triggers: List[str] = []
    decision = "different"

    if cfg.triggers.get("MERGE_AI_ON_BALOWED_EXACT", True) and field_matches.get(
        "balance_owed"
    ):
        triggers.append("strong:balance_owed")
        if decision == "different":
            decision = "ai"

    acctnum_aux = aux.get("account_number", {})
    acct_level = str(acctnum_aux.get("acctnum_level") or "")
    min_level = str(cfg.triggers.get("MERGE_AI_ON_ACCTNUM_LEVEL", "last4")).lower()
    if field_matches.get("account_number") and acct_level:
        if _ACCOUNT_LEVEL_ORDER.get(acct_level, 0) >= _ACCOUNT_LEVEL_ORDER.get(
            min_level, 0
        ) and acct_level != "none":
            triggers.append("strong:account_number")
            if decision == "different":
                decision = "ai"

    mid_threshold = int(cfg.triggers.get("MERGE_AI_ON_MID_K", 0))
    if mid_sum >= mid_threshold and mid_threshold > 0:
        triggers.append("mid")
        if decision == "different":
            decision = "ai"

    if cfg.triggers.get("MERGE_AI_ON_ALL_DATES", False) and dates_all:
        triggers.append("dates")
        if decision == "different":
            decision = "ai"

    ai_threshold = int(cfg.thresholds.get("AI_THRESHOLD", 0))
    if total >= ai_threshold and ai_threshold > 0:
        triggers.append("total")
        if decision == "different":
            decision = "ai"

    conflicts: List[str] = []
    last4_a = _collect_account_last4(A_data)
    last4_b = _collect_account_last4(B_data)
    if last4_a and last4_b and last4_a.isdisjoint(last4_b):
        conflicts.append("acct_last4_mismatch")

    for conflict in _detect_amount_conflicts(A_data, B_data, cfg):
        if conflict not in conflicts:
            conflicts.append(conflict)

    auto_threshold = int(cfg.thresholds.get("AUTO_MERGE_THRESHOLD", 0))
    if total >= auto_threshold and auto_threshold > 0 and not conflicts:
        decision = "auto"

    result = {
        "total": int(total),
        "parts": parts,
        "mid_sum": int(mid_sum),
        "dates_all": dates_all,
        "aux": aux,
        "triggers": triggers,
        "conflicts": conflicts,
        "decision": decision,
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


def score_all_pairs(
    sid: str,
    idx_list: Iterable[int],
    runs_root: Path = Path("runs"),
) -> Dict[int, Dict[int, Dict[str, Any]]]:
    """Score all unordered account pairs for a case run."""

    cfg = get_merge_cfg()

    indices: List[int] = []
    for raw_idx in idx_list:
        if isinstance(raw_idx, bool):
            continue
        try:
            idx_val = int(raw_idx)
        except (TypeError, ValueError):
            logger.warning("MERGE_SCORE sid=<%s> invalid_index=%r", sid, raw_idx)
            continue
        indices.append(idx_val)

    unique_indices = sorted(set(indices))
    bureaus_by_idx: Dict[int, Dict[str, Dict[str, Any]]] = {}
    for idx in unique_indices:
        try:
            bureaus = load_bureaus(sid, idx, runs_root=runs_root)
        except FileNotFoundError:
            logger.warning(
                "MERGE_SCORE sid=<%s> idx=<%s> bureaus_missing", sid, idx
            )
            bureaus = {}
        except Exception:
            logger.exception(
                "MERGE_SCORE sid=<%s> idx=<%s> bureaus_load_failed", sid, idx
            )
            bureaus = {}
        bureaus_by_idx[idx] = bureaus

    scores: Dict[int, Dict[int, Dict[str, Any]]] = {
        idx: {} for idx in unique_indices
    }

    for left, right in combinations(unique_indices, 2):
        result = score_pair(bureaus_by_idx.get(left, {}), bureaus_by_idx.get(right, {}), cfg)
        logger.info(
            "MERGE_SCORE sid=<%s> pair=(%s,%s) total=<%s> decision=<%s>",
            sid,
            left,
            right,
            result.get("total"),
            result.get("decision"),
        )
        scores[left][right] = deepcopy(result)
        scores[right][left] = deepcopy(result)

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
    acct_level = "none"
    by_field_pairs: Dict[str, List[str]] = {}
    matched_fields: Dict[str, bool] = {}

    if isinstance(aux, Mapping):
        acct_aux = aux.get("account_number")
        if isinstance(acct_aux, Mapping):
            level = acct_aux.get("acctnum_level")
            if isinstance(level, str) and level:
                acct_level = level

        for field in _FIELD_SEQUENCE:
            field_aux = aux.get(field) if isinstance(aux, Mapping) else None
            if not isinstance(field_aux, Mapping):
                continue
            if "matched" in field_aux:
                matched_fields[field] = bool(field_aux.get("matched"))
            best_pair = field_aux.get("best_pair")
            if best_pair and isinstance(best_pair, (list, tuple)) and len(best_pair) == 2:
                by_field_pairs[field] = [str(best_pair[0]), str(best_pair[1])]

    return {
        "acctnum_level": acct_level,
        "by_field_pairs": by_field_pairs,
        "matched_fields": matched_fields,
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
        return merge_tag

    score_total = int(best_result.get("total", 0) or 0)
    decision = str(best_result.get("decision", "different"))
    triggers = list(best_result.get("triggers", []))
    conflicts = list(best_result.get("conflicts", []))
    parts = _sanitize_parts(best_result.get("parts"))
    aux_payload = _build_aux_payload(best_result.get("aux", {}))

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
    for idx in all_indices:
        partner_scores = scores_by_idx.get(idx, {})
        best_info = best_by_idx.get(idx, {})
        merge_tag = _merge_tag_from_best(idx, partner_scores, best_info)
        merge_tags[idx] = merge_tag
        _persist_merge_tag(sid, idx, merge_tag, runs_root, None)
        logger.info(
            "MERGE_TAG sid=<%s> idx=<%s> decision=<%s> score=<%s> tiebreaker=<%s>",
            sid,
            idx,
            merge_tag.get("decision"),
            merge_tag.get("score_total"),
            merge_tag.get("tiebreaker"),
        )

    return merge_tags


def score_and_tag_best_partners(
    sid: str,
    idx_list: Iterable[int],
    runs_root: Path = Path("runs"),
) -> Dict[int, Dict[str, Any]]:
    """Convenience wrapper to score accounts, pick best partners, and persist tags."""

    scores = score_all_pairs(sid, idx_list, runs_root=runs_root)
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
    return digits_only(value)


def account_number_level(a: Any, b: Any) -> str:
    digits_a = digits_only(a)
    digits_b = digits_only(b)
    if not digits_a or not digits_b:
        return "none"

    norm_a = digits_a.lstrip("0") or "0"
    norm_b = digits_b.lstrip("0") or "0"
    if norm_a == norm_b:
        return "exact"

    if len(digits_a) >= 4 and len(digits_b) >= 4:
        if digits_a[-4:] == digits_b[-4:]:
            return "last4"

    return "none"


def account_numbers_match(a: Any, b: Any, min_level: str = "last4") -> Tuple[bool, str]:
    level = account_number_level(a, b)
    threshold = _ACCOUNT_LEVEL_ORDER.get(min_level, 0)
    match = _ACCOUNT_LEVEL_ORDER.get(level, 0) >= threshold and level != "none"
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

    # Remove trailing time components if present (e.g., ISO timestamps).
    if "T" in text:
        text = text.split("T", 1)[0]
    if " " in text and len(text.split()) > 1:
        text = text.split()[0]

    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue

    # Attempt separator normalization and retry common Y/M/D patterns.
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
    delta = abs((a - b).days)
    return delta <= days


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

@dataclass
class LegacyMergeCfg:
    """Legacy configuration used by the existing scoring pipeline."""

    weights: Dict[str, float]
    thresholds: Dict[str, float]
    acctnum_trigger_ai: str = "any"
    acctnum_min_score: float = 0.31
    acctnum_require_masked: bool = False
    balowed_tol_abs: float = 0.0
    balowed_tol_ratio: float = 0.0


_DEFAULT_THRESHOLDS = {
    "auto_merge_min": 0.78,
    "ai_band_min": 0.35,
    "ai_band_max": 0.78,
    "ai_hard_min": 0.30,
}

_DEFAULT_WEIGHTS = {
    "balowed": 0.25,
    "acct_num": 0.25,
    "dates": 0.20,
    "status": 0.20,
    "strings": 0.10,
}

_DEFAULT_ACCTNUM_TRIGGER = "any"
_DEFAULT_ACCTNUM_MIN_SCORE = 0.31
_DEFAULT_ACCTNUM_REQUIRE_MASKED = False

_ENV_THRESHOLD_KEYS = {
    "auto_merge_min": "MERGE_AUTO_MIN",
    "ai_band_min": "MERGE_AI_MIN",
    "ai_band_max": "MERGE_AI_MAX",
    "ai_hard_min": "MERGE_AI_HARD_MIN",
}

_ENV_WEIGHT_KEYS = {
    "balowed": "MERGE_W_BALOWED",
    "acct_num": "MERGE_W_ACCT",
    "dates": "MERGE_W_DATES",
    "status": "MERGE_W_STATUS",
    "strings": "MERGE_W_STRINGS",
}

_BALOWED_TOL_ABS_KEY = "MERGE_BALOWED_TOL_ABS"
_BALOWED_TOL_RATIO_KEY = "MERGE_BALOWED_TOL_RATIO"
_BALOWED_TRIGGER_KEY = "MERGE_BALOWED_TRIGGER_AI"
_BALOWED_MIN_SCORE_KEY = "MERGE_BALOWED_MIN_SCORE"

_ACCTNUM_TRIGGER_CHOICES = {"off", "exact", "last4", "masked", "any"}
_ACCTNUM_TRIGGER_KEY = "MERGE_ACCTNUM_TRIGGER_AI"
_ACCTNUM_MIN_SCORE_KEY = "MERGE_ACCTNUM_MIN_SCORE"
_ACCTNUM_REQUIRE_MASKED_KEY = "MERGE_ACCTNUM_REQUIRE_MASKED"

_ACCT_NUMBER_FIELDS = (
    "account_number",
    "acct_num",
    "number",
    "account_number_display",
)
_MASKED_ACCOUNT_PATTERN = re.compile(r"[xX*#•]")


def _normalized_mask_skeleton(value: Optional[str]) -> Optional[str]:
    """Collapse masking characters into a normalized skeleton string."""

    if not value:
        return None

    mask_chars = _MASKED_ACCOUNT_PATTERN.findall(value)
    if not mask_chars:
        return None

    return "x" * len(mask_chars)


def _read_json(p: Path) -> Union[Dict[str, Any], List[Any]]:
    if not p.exists():
        return {}
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def _extract_acct_from_raw_lines(
    raw_lines: Optional[List[Dict[str, Any]]]
) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    if not raw_lines:
        return None, None
    for row in raw_lines:
        text = (row.get("text") or "").strip()
        if text.startswith("Account #"):
            parts = text.replace("Account #", "").strip().split()
            value = parts[0] if parts else None
            return value, row
    return None, None


def _manifest_group_key(acc_idx: int) -> str:
    return f"cases.accounts.{acc_idx}"


def _manifest_path(
    manifest: Optional[RunManifest], group: str, key: str
) -> Optional[Path]:
    if manifest is None:
        return None
    try:
        return Path(manifest.get(group, key))
    except KeyError:
        return None


def load_case_account(
    sid: str,
    acc_idx: int,
    runs_root: Path,
    manifest: Optional[RunManifest] = None,
) -> Dict[str, Any]:
    group = _manifest_group_key(acc_idx)
    base = _manifest_path(manifest, group, "dir")
    if base is None:
        base = runs_root / sid / "cases" / "accounts" / str(acc_idx)
        if not base.exists():
            alt_base = runs_root / "cases" / sid / "accounts" / str(acc_idx)
            if alt_base.exists():
                base = alt_base
    fields_path = _manifest_path(manifest, group, "flat") or (base / "fields_flat.json")
    bureaus_path = _manifest_path(manifest, group, "bureaus") or (base / "bureaus.json")
    raw_path = _manifest_path(manifest, group, "raw") or (base / "raw_lines.json")

    if not base.exists():
        raise FileNotFoundError(
            f"case data not found for sid={sid!r} index={acc_idx} under {runs_root}"
        )

    fields = _read_json(fields_path)
    bureaus = _read_json(bureaus_path)
    raw = _read_json(raw_path)

    balowed = fields.get("balance_owed")
    if balowed in (None, "", "--"):
        balowed = fields.get("past_due_amount")

    status = fields.get("payment_status") or fields.get("account_status")
    dates = {
        "opened": fields.get("date_opened"),
        "dla": fields.get("date_of_last_activity"),
        "closed": fields.get("closed_date"),
        "reported": fields.get("date_reported"),
    }

    acct_display = (
        (bureaus.get("transunion") or {}).get("account_number")
        or (bureaus.get("experian") or {}).get("account_number")
        or (bureaus.get("equifax") or {}).get("account_number")
    )
    if not acct_display:
        acct_display, _ = _extract_acct_from_raw_lines(
            raw if isinstance(raw, list) else []
        )

    return {
        "index": acc_idx,
        "fields_flat": fields,
        "bureaus": bureaus,
        "raw_lines": raw,
        "norm": {
            "balance_owed": balowed,
            "status": status,
            "dates": dates,
            "account_number_display": acct_display,
        },
        "case_dir": base,
    }


def _read_env_int(env: Mapping[str, str], key: str, default: int) -> int:
    value = env.get(key)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        logger.warning(
            "Invalid integer for %s=%r; falling back to default %s",
            key,
            value,
            default,
        )
        return default


def _read_env_float(env: Mapping[str, str], key: str, default: float) -> float:
    value = env.get(key)
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError:
        logger.warning(
            "Invalid float for %s=%r; falling back to default %.4f", key, value, default
        )
        return default


def _read_env_choice(
    env: Mapping[str, str], key: str, default: str, choices: Set[str]
) -> str:
    value = env.get(key)
    if value is None or value == "":
        return default
    normalized = str(value).strip().lower()
    if normalized not in choices:
        logger.warning(
            "Invalid option for %s=%r; falling back to default %s", key, value, default
        )
        return default
    return normalized


def _read_env_flag(env: Mapping[str, str], key: str, default: bool) -> bool:
    value = env.get(key)
    if value is None or value == "":
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        logger.warning(
            "Invalid flag for %s=%r; falling back to default %s", key, value, int(default)
        )
        return default
    if parsed not in (0, 1):
        logger.warning(
            "Invalid flag for %s=%r; falling back to default %s", key, value, int(default)
        )
        return default
    return bool(parsed)


def load_config_from_env(env: Optional[Mapping[str, str]] = None) -> LegacyMergeCfg:
    """Create a LegacyMergeCfg using optional environment overrides."""

    env_mapping: Mapping[str, str]
    if env is None:
        env_mapping = os.environ
    else:
        env_mapping = env

    thresholds = {
        name: _read_env_float(env_mapping, env_key, _DEFAULT_THRESHOLDS[name])
        for name, env_key in _ENV_THRESHOLD_KEYS.items()
    }

    weights = {
        name: _read_env_float(env_mapping, env_key, _DEFAULT_WEIGHTS[name])
        for name, env_key in _ENV_WEIGHT_KEYS.items()
    }

    balowed_tol_abs = _read_env_float(env_mapping, _BALOWED_TOL_ABS_KEY, 0.0)
    balowed_tol_ratio = _read_env_float(env_mapping, _BALOWED_TOL_RATIO_KEY, 0.0)

    acctnum_trigger_ai = _read_env_choice(
        env_mapping, _ACCTNUM_TRIGGER_KEY, _DEFAULT_ACCTNUM_TRIGGER, _ACCTNUM_TRIGGER_CHOICES
    )
    acctnum_min_score = _read_env_float(
        env_mapping, _ACCTNUM_MIN_SCORE_KEY, _DEFAULT_ACCTNUM_MIN_SCORE
    )
    acctnum_require_masked = _read_env_flag(
        env_mapping, _ACCTNUM_REQUIRE_MASKED_KEY, _DEFAULT_ACCTNUM_REQUIRE_MASKED
    )

    return LegacyMergeCfg(
        weights=weights,
        thresholds=thresholds,
        acctnum_trigger_ai=acctnum_trigger_ai,
        acctnum_min_score=acctnum_min_score,
        acctnum_require_masked=acctnum_require_masked,
        balowed_tol_abs=balowed_tol_abs,
        balowed_tol_ratio=balowed_tol_ratio,
    )


DEFAULT_CFG: LegacyMergeCfg = load_config_from_env()

_DATE_FIELDS = ("date_opened", "date_of_last_activity", "closed_date")
_STATUS_FIELDS = ("payment_status", "account_status")
_STRING_FIELDS = ("creditor", "remarks")
_STATUS_PATTERNS: Tuple[Tuple[str, re.Pattern[str]], ...] = (
    ("collection", re.compile(r"collection|charge\s*off|repo|repossession|foreclosure", re.I)),
    ("delinquent", re.compile(r"delinquent|late|past\s+due|overdue|derog", re.I)),
    ("paid", re.compile(r"paid\b|paid\s+off|settled|resolved|pif", re.I)),
    (
        "current",
        re.compile(r"current|pays\s+as\s+agreed|paid\s+as\s+agreed|open", re.I),
    ),
    ("closed", re.compile(r"closed|inactive|terminated", re.I)),
    ("bankruptcy", re.compile(r"bankrupt|bk", re.I)),
)


def _normalize_account_number(value: Any) -> Optional[str]:
    digits = re.sub(r"\D", "", str(value or ""))
    return digits or None


def _prepare_account_for_scoring(
    case_account: Mapping[str, Any]
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    fields = dict(case_account.get("fields_flat") or {})
    bureaus = case_account.get("bureaus") or {}
    norm = case_account.get("norm") or {}

    prepared: Dict[str, Any] = {}
    meta: Dict[str, Any] = {}

    # Start with flattened fields to retain as much context as possible.
    for key, value in fields.items():
        prepared[key] = value

    balowed = norm.get("balance_owed")
    if balowed not in (None, "", "--"):
        prepared["balance_owed"] = balowed

    balance_value = _parse_currency(balowed)
    if balance_value is None:
        balance_value = _parse_currency(fields.get("balance_owed"))
    if balance_value is None:
        balance_value = _parse_currency(fields.get("past_due_amount"))
    if balance_value is not None:
        meta["balance_owed"] = balance_value

    status = norm.get("status")
    if status not in (None, ""):
        prepared["payment_status"] = status
        prepared.setdefault("account_status", status)

    dates = norm.get("dates") or {}
    meta_dates: Dict[str, Any] = {}
    if isinstance(dates, Mapping):
        date_map = {
            "date_opened": dates.get("opened"),
            "date_of_last_activity": dates.get("dla"),
            "closed_date": dates.get("closed"),
            "date_reported": dates.get("reported"),
        }
        for key, value in date_map.items():
            if value not in (None, ""):
                prepared[key] = value
            if value not in (None, ""):
                meta_dates[key] = value
    if meta_dates:
        meta["dates"] = meta_dates

    acct_display = norm.get("account_number_display")
    acct_display_source: Optional[str] = None
    acct_display_raw_line: Optional[Dict[str, Any]] = None
    if acct_display not in (None, "", "--"):
        acct_display_source = "norm"
    if acct_display in (None, "", "--"):
        for bureau_key in ("transunion", "experian", "equifax"):
            bureau_val = (bureaus.get(bureau_key) or {}).get("account_number")
            if bureau_val not in (None, ""):
                acct_display = bureau_val
                acct_display_source = f"bureau:{bureau_key}"
                break
    if acct_display in (None, "", "--"):
        raw_lines = case_account.get("raw_lines")
        if isinstance(raw_lines, list):
            acct_display, raw_entry = _extract_acct_from_raw_lines(raw_lines)
            if acct_display not in (None, ""):
                acct_display_source = "raw_lines"
                acct_display_raw_line = raw_entry
    if acct_display not in (None, ""):
        prepared["account_number"] = acct_display
        prepared["acct_num"] = acct_display
        prepared["number"] = acct_display
        prepared["account_number_display"] = acct_display
        meta["account_number_display"] = acct_display
    if acct_display_source:
        meta["acct_display_source"] = acct_display_source
    if acct_display_raw_line:
        meta["acct_display_raw_line"] = acct_display_raw_line

    if "creditor" not in prepared or not str(prepared.get("creditor") or "").strip():
        for key in ("creditor", "creditor_name", "original_creditor", "creditor_type"):
            value = fields.get(key) or (bureaus.get("transunion") or {}).get(key)
            if value not in (None, "", "--"):
                prepared["creditor"] = value
                break

    if "remarks" not in prepared or not str(prepared.get("remarks") or "").strip():
        for key in ("remarks", "creditor_remarks", "comment", "comments"):
            value = fields.get(key) or (bureaus.get("transunion") or {}).get(key)
            if value not in (None, "", "--"):
                prepared["remarks"] = value
                break

    account_index = case_account.get("index")
    if isinstance(account_index, int):
        prepared.setdefault("account_index", account_index)
        meta["account_index"] = account_index

    return prepared, meta


def _extract_account_number_fields(account: Mapping[str, Any]) -> Dict[str, Any]:
    raw: Optional[str] = None
    for key in _ACCT_NUMBER_FIELDS:
        if key in account and account.get(key) not in (None, ""):
            raw = str(account.get(key))
            if raw.strip():
                break
    digits = _normalize_account_number(raw)
    last4 = digits[-4:] if digits and len(digits) >= 4 else None
    masked = bool(raw and _MASKED_ACCOUNT_PATTERN.search(raw))
    skeleton = _normalized_mask_skeleton(raw)
    return {
        "acct_num_raw": raw,
        "acct_num_digits": digits,
        "acct_num_masked": masked,
        "acct_num_last4": last4,
        "acct_num_mask_skeleton": skeleton,
    }


def acctnum_match_level(a: Mapping[str, Any], b: Mapping[str, Any]) -> str:
    digits_a = (a or {}).get("acct_num_digits") or ""
    digits_b = (b or {}).get("acct_num_digits") or ""
    if digits_a and digits_b and digits_a == digits_b:
        return "exact"

    last4_a = (a or {}).get("acct_num_last4") or ""
    last4_b = (b or {}).get("acct_num_last4") or ""
    if last4_a and last4_b and last4_a == last4_b:
        return "last4"

    masked_a = bool((a or {}).get("acct_num_masked"))
    masked_b = bool((b or {}).get("acct_num_masked"))
    if masked_a and masked_b and not (digits_a or digits_b):
        skeleton_a = (a or {}).get("acct_num_mask_skeleton") or ""
        skeleton_b = (b or {}).get("acct_num_mask_skeleton") or ""
        if skeleton_a and skeleton_a == skeleton_b:
            return "masked"

    return "none"


def _score_account_number(acc_a: Dict[str, Any], acc_b: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
    norm_a = _extract_account_number_fields(acc_a)
    norm_b = _extract_account_number_fields(acc_b)
    level = acctnum_match_level(norm_a, norm_b)
    if level == "exact":
        score = 1.0
    elif level == "last4":
        score = 0.7
    else:
        score = 0.0
    masked_any = bool(norm_a.get("acct_num_masked") or norm_b.get("acct_num_masked"))
    aux = {
        "acctnum_level": level,
        "acctnum_masked_any": masked_any,
        "acct_num_a": norm_a,
        "acct_num_b": norm_b,
    }
    return score, aux


def _parse_date(value: Any) -> Optional[date]:
    if not value:
        return None
    text = str(value).strip()
    if not text:
        return None
    text = text.replace("/", ".").replace("-", ".")
    parts = text.split(".")
    if len(parts) != 3:
        return None
    try:
        day, month, year = (int(part) for part in parts)
        return date(year, month, day)
    except ValueError:
        return None


def _score_date_values(dates_a: Iterable[Optional[date]], dates_b: Iterable[Optional[date]]) -> float:
    scores: List[float] = []
    for da, db in zip(dates_a, dates_b):
        if da and db:
            delta = abs((da - db).days)
            scaled = max(0.0, 1.0 - min(delta, 365) / 365.0)
            scores.append(scaled)
    if not scores:
        return 0.0
    return sum(scores) / len(scores)


def _score_dates(acc_a: Dict[str, Any], acc_b: Dict[str, Any]) -> float:
    values_a = [_parse_date(acc_a.get(field)) for field in _DATE_FIELDS]
    values_b = [_parse_date(acc_b.get(field)) for field in _DATE_FIELDS]
    return _score_date_values(values_a, values_b)


def _parse_currency(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    text = text.replace(",", "").replace("$", "").replace("USD", "")
    if text in {"-", "n/a", "na"}:
        return None
    try:
        return float(text)
    except ValueError:
        # Attempt to extract digits (including decimal point)
        match = re.search(r"-?\d+(?:\.\d+)?", text)
        if not match:
            return None
        try:
            return float(match.group())
        except ValueError:
            return None


def _normalized_balance_owed(account: Mapping[str, Any]) -> Optional[float]:
    norm_block = account.get("norm")
    if isinstance(norm_block, Mapping):
        parsed = _parse_currency(norm_block.get("balance_owed"))
        if parsed is not None:
            return parsed

    for key in (
        "balance_owed_norm",
        "balance_owed_normalized",
        "normalized_balance_owed",
    ):
        parsed = _parse_currency(account.get(key))
        if parsed is not None:
            return parsed

    return _parse_currency(account.get("balance_owed"))


def _score_balowed_values(
    balance_a: Optional[float],
    balance_b: Optional[float],
    tol_abs: float,
    tol_ratio: float,
) -> float:
    if balance_a is None or balance_b is None:
        return 0.0

    delta = abs(balance_a - balance_b)
    base = max(abs(balance_a), abs(balance_b))

    tol_abs = max(tol_abs, 0.0)
    tol_ratio = max(tol_ratio, 0.0)
    allowed = max(tol_abs, tol_ratio * base)

    if delta <= allowed:
        return 1.0

    denom = max(base, 1.0)
    over = delta - allowed
    scaled = 1.0 - min(over / denom, 1.0)
    return max(0.0, scaled)


def _clean_status(value: Any) -> Optional[str]:
    if not value:
        return None
    text = str(value).lower()
    text = text.replace("/", " ").replace("-", " ")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text or None


def _status_buckets_from_text(text: str) -> List[str]:
    buckets = [name for name, pattern in _STATUS_PATTERNS if pattern.search(text)]
    if not buckets and text:
        buckets.append("other:" + text)
    return buckets


def _score_status(acc_a: Dict[str, Any], acc_b: Dict[str, Any]) -> float:
    buckets_a: set[str] = set()
    buckets_b: set[str] = set()
    for field in _STATUS_FIELDS:
        norm_a = _clean_status(acc_a.get(field))
        norm_b = _clean_status(acc_b.get(field))
        if norm_a:
            buckets_a.update(_status_buckets_from_text(norm_a))
        if norm_b:
            buckets_b.update(_status_buckets_from_text(norm_b))
    if not buckets_a or not buckets_b:
        return 0.0
    if buckets_a & buckets_b:
        return 1.0
    return 0.0


def _normalize_text_fields(account: Dict[str, Any]) -> str:
    parts: List[str] = []
    for field in _STRING_FIELDS:
        value = account.get(field)
        if value:
            cleaned = re.sub(r"\s+", " ", str(value).lower()).strip()
            if cleaned:
                parts.append(cleaned)
    return " ".join(parts)


def _score_strings(acc_a: Dict[str, Any], acc_b: Dict[str, Any]) -> float:
    text_a = _normalize_text_fields(acc_a)
    text_b = _normalize_text_fields(acc_b)
    if not text_a or not text_b:
        return 0.0
    return difflib.SequenceMatcher(None, text_a, text_b).ratio()


def score_accounts(
    accA: Dict[str, Any], accB: Dict[str, Any], cfg: LegacyMergeCfg = DEFAULT_CFG
) -> Tuple[float, Dict[str, float], Dict[str, Any]]:
    """Return overall score, per-part contributions, and auxiliary details."""

    acct_score, acct_aux = _score_account_number(accA, accB)
    balance_a = _normalized_balance_owed(accA)
    balance_b = _normalized_balance_owed(accB)
    balowed_score = _score_balowed_values(
        balance_a,
        balance_b,
        getattr(cfg, "balowed_tol_abs", 0.0),
        getattr(cfg, "balowed_tol_ratio", 0.0),
    )

    parts = {
        "acct_num": acct_score,
        "dates": _score_dates(accA, accB),
        "balowed": balowed_score,
        "status": _score_status(accA, accB),
        "strings": _score_strings(accA, accB),
    }

    weights_present = {
        name: cfg.weights.get(name, 0.0)
        for name in parts
        if cfg.weights.get(name, 0.0) > 0.0
    }
    total_weight = sum(weights_present.values())
    if total_weight <= 0:
        aux = dict(acct_aux)
        if balance_a is not None:
            aux["balowed_a"] = balance_a
        if balance_b is not None:
            aux["balowed_b"] = balance_b
        return 0.0, parts, aux

    weighted = sum(parts[name] * weight for name, weight in weights_present.items())
    score = weighted / total_weight

    aux = dict(acct_aux)
    if balance_a is not None:
        aux["balowed_a"] = balance_a
    if balance_b is not None:
        aux["balowed_b"] = balance_b

    return score, parts, aux


def decide_merge(score: float, cfg: LegacyMergeCfg = DEFAULT_CFG) -> str:
    """Return decision label based on configured thresholds."""

    thresholds = cfg.thresholds
    auto_min = thresholds.get("auto_merge_min", 0.78)
    ai_min = thresholds.get("ai_band_min", 0.35)
    ai_max = thresholds.get("ai_band_max", auto_min)
    hard_min = thresholds.get("ai_hard_min", ai_min)

    if score >= auto_min:
        return "auto"
    if score < hard_min:
        return "different"
    if ai_min <= score < ai_max:
        return "ai"
    if score >= ai_max:
        return "ai"
    return "different"


def _maybe_apply_balowed_override(
    parts: Mapping[str, float],
    score: float,
    cfg: LegacyMergeCfg,
    meta: Mapping[str, Any],
) -> Tuple[float, Optional[str], Dict[str, Any]]:
    """Lift low scores into the AI band when balances match perfectly."""

    trigger_enabled = _read_env_flag(os.environ, _BALOWED_TRIGGER_KEY, False)
    if not trigger_enabled:
        return score, None, {}

    balowed_part = parts.get("balowed", 0.0)
    if balowed_part < 0.9999:
        return score, None, {}

    thresholds = cfg.thresholds or {}
    ai_min = thresholds.get("ai_band_min", 0.35)
    if score >= ai_min:
        return score, None, {}

    min_score = _read_env_float(os.environ, _BALOWED_MIN_SCORE_KEY, 0.31)
    if score < min_score:
        return score, None, {}

    aux = dict(meta.get("aux") or {})
    balance_a = aux.get("balowed_a")
    balance_b = aux.get("balowed_b")

    tol_abs = max(getattr(cfg, "balowed_tol_abs", 0.0), 0.0)
    tol_ratio = max(getattr(cfg, "balowed_tol_ratio", 0.0), 0.0)

    delta: Optional[float] = None
    allowed: float = tol_abs
    exact = False

    if isinstance(balance_a, (int, float)) and isinstance(balance_b, (int, float)):
        a_val = float(balance_a)
        b_val = float(balance_b)
        delta = abs(a_val - b_val)
        base = max(abs(a_val), abs(b_val))
        allowed = max(tol_abs, tol_ratio * base)
        exact = delta <= 1e-6
    else:
        exact = balowed_part >= 0.9999 and tol_abs <= 0 and tol_ratio <= 0

    reasons: Dict[str, Any] = {
        "balance_only_triggers_ai": True,
        "balance_exact_match": bool(exact),
    }
    if delta is not None:
        reasons["balance_delta"] = float(delta)
        reasons["balance_tolerance"] = float(allowed)

    sid_value = meta.get("sid") or "-"
    idx_a = meta.get("idx_a")
    idx_b = meta.get("idx_b")

    logger.info(
        "MERGE_OVERRIDE sid=<%s> i=<%s> j=<%s> kind=balowed exact=<%d> delta=<%.4f> "
        "allowed=<%.4f> score=<%.4f>",
        sid_value,
        idx_a if isinstance(idx_a, int) else "-",
        idx_b if isinstance(idx_b, int) else "-",
        1 if exact else 0,
        float(delta) if delta is not None else 0.0,
        allowed,
        score,
    )

    return score, "ai", reasons


def _apply_account_number_ai_override(
    base_score: float,
    score: float,
    decision: str,
    aux: Dict[str, Any],
    reasons: Dict[str, Any],
    thresholds: Mapping[str, float],
    *,
    sid_value: str,
    idx_a: int,
    idx_b: int,
) -> Tuple[float, str, Dict[str, Any], Dict[str, Any]]:
    """Lift low scores into the AI band for strong account-number matches."""

    ai_band_min = thresholds.get("ai_band_min", 0.35)
    ai_hard_min = thresholds.get("ai_hard_min", ai_band_min)

    level = aux.get("acctnum_level") or "none"
    masked_any = bool(aux.get("acctnum_masked_any", False))

    env = os.environ
    trig = _read_env_choice(
        env, _ACCTNUM_TRIGGER_KEY, _DEFAULT_ACCTNUM_TRIGGER, _ACCTNUM_TRIGGER_CHOICES
    )
    minscore = _read_env_float(env, _ACCTNUM_MIN_SCORE_KEY, _DEFAULT_ACCTNUM_MIN_SCORE)
    req_masked = _read_env_flag(
        env, _ACCTNUM_REQUIRE_MASKED_KEY, _DEFAULT_ACCTNUM_REQUIRE_MASKED
    )

    if trig == "off":
        eligible = False
    elif trig == "exact":
        eligible = level == "exact"
    elif trig == "last4":
        eligible = level in {"exact", "last4"}
    elif trig == "masked":
        eligible = level in {"exact", "last4", "masked"}
    else:  # any
        eligible = level in {"exact", "last4", "masked"}
    if req_masked:
        eligible = eligible and masked_any

    if eligible and base_score < ai_band_min:
        lifted_score = max(score, minscore, ai_hard_min)
        updated_reasons = dict(reasons or {})
        reason_list: List[Dict[str, Any]]
        existing = updated_reasons.get("reasons")
        if isinstance(existing, list):
            reason_list = list(existing)
        else:
            reason_list = []
        reason_entry = {"kind": "acctnum", "level": level, "masked_any": masked_any}
        reason_list.append(reason_entry)

        updated_reasons["reasons"] = reason_list
        updated_reasons["acctnum_only_triggers_ai"] = True
        updated_reasons["acctnum_match_level"] = level
        updated_reasons["acctnum_masked_any"] = masked_any

        aux_with_reasons = dict(aux)
        aux_with_reasons["override_reasons"] = dict(updated_reasons)
        aux_with_reasons["override_reason_entries"] = list(reason_list)

        logger.info(
            "MERGE_OVERRIDE sid=<%s> i=<%d> j=<%d> kind=acctnum level=<%s> masked_any=<%d> lifted_to=<%.4f>",
            sid_value,
            idx_a,
            idx_b,
            level or "none",
            1 if masked_any else 0,
            lifted_score,
        )

        return lifted_score, "ai", updated_reasons, aux_with_reasons

    if reasons:
        aux = dict(aux)
        aux["override_reasons"] = dict(reasons)
        if isinstance(reasons.get("reasons"), list):
            aux["override_reason_entries"] = list(reasons["reasons"])
    return score, decision, reasons, aux


def _build_ai_side_snapshot(
    account: Mapping[str, Any], acct_meta: Optional[Mapping[str, Any]]
) -> Dict[str, Any]:
    snapshot: Dict[str, Any] = {}

    idx = account.get("account_index")
    if isinstance(idx, int):
        snapshot["account_index"] = idx

    account_id = account.get("account_id")
    if account_id is not None:
        snapshot["account_id"] = account_id

    highlights: Dict[str, Any] = {}
    balance_val = _parse_currency(account.get("balance_owed"))
    if balance_val is not None:
        highlights["balance_owed"] = balance_val

    if acct_meta:
        raw_val = acct_meta.get("acct_num_raw")
        if raw_val not in (None, ""):
            highlights["acct_num_raw"] = raw_val
        last4 = acct_meta.get("acct_num_last4")
        if last4 not in (None, ""):
            highlights["acct_num_last4"] = last4

    if highlights:
        snapshot["highlights"] = highlights

    return snapshot


def build_ai_decision_pack(
    left_account: Mapping[str, Any],
    right_account: Mapping[str, Any],
    *,
    score: float,
    parts: Mapping[str, float],
    aux: Mapping[str, Any],
    reasons: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Assemble an AI review pack payload for a matched account pair."""

    aux_dict = dict(aux or {})
    pack: Dict[str, Any] = {
        "decision": "ai",
        "score": float(score),
        "parts": {name: float(value) for name, value in dict(parts or {}).items()},
        "left": _build_ai_side_snapshot(left_account, aux_dict.get("acct_num_a")),
        "right": _build_ai_side_snapshot(right_account, aux_dict.get("acct_num_b")),
        "acctnum": {
            "level": aux_dict.get("acctnum_level") or "none",
            "masked_any": bool(aux_dict.get("acctnum_masked_any", False)),
        },
    }

    if reasons:
        if isinstance(reasons, Mapping):
            payload = reasons.get("reasons")
            if isinstance(payload, list):
                pack["reasons"] = [dict(item) for item in payload]
            else:
                pack["reasons"] = dict(reasons)
        elif isinstance(reasons, list):
            pack["reasons"] = [dict(item) for item in reasons]

    return pack


def _build_auto_graph(size: int, auto_edges: List[Tuple[int, int]]) -> List[List[int]]:
    graph: List[List[int]] = [[] for _ in range(size)]
    for i, j in auto_edges:
        graph[i].append(j)
        graph[j].append(i)
    return graph


def _connected_components(graph: List[List[int]]) -> List[List[int]]:
    visited = [False] * len(graph)
    components: List[List[int]] = []

    for idx in range(len(graph)):
        if visited[idx]:
            continue
        stack = [idx]
        component: List[int] = []
        visited[idx] = True
        while stack:
            node = stack.pop()
            component.append(node)
            for neighbor in graph[node]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    stack.append(neighbor)
        components.append(sorted(component))
    return components


def _persist_merge_tag(
    sid: str,
    account_index: Optional[int],
    merge_tag: Mapping[str, Any],
    runs_root_path: Optional[Path],
    manifest: Optional[RunManifest],
) -> None:
    if runs_root_path is None or not sid or not isinstance(account_index, int):
        return

    summary_path = _manifest_path(
        manifest, _manifest_group_key(account_index), "summary"
    )
    if summary_path is None:
        summary_path = (
            runs_root_path
            / sid
            / "cases"
            / "accounts"
            / str(account_index)
            / "summary.json"
        )

    summary_dir = summary_path.parent
    if not summary_dir.exists():
        return

    summary_obj: Dict[str, Any] = {}
    if summary_path.exists():
        try:
            summary_obj = json.loads(summary_path.read_text(encoding="utf-8"))
            if not isinstance(summary_obj, dict):
                summary_obj = {}
        except Exception:
            summary_obj = {}

    try:
        sanitized = json.loads(json.dumps(merge_tag, ensure_ascii=False))
    except TypeError:
        sanitized = dict(merge_tag)

    summary_obj["merge_tag"] = sanitized

    try:
        summary_path.write_text(
            json.dumps(summary_obj, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception:
        logger.exception(
            "MERGE_SCORE sid=<%s> summary_write_failed path=%s", sid, summary_path
        )


def _sanitize_for_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _sanitize_for_json(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_sanitize_for_json(item) for item in value]
    if isinstance(value, tuple):
        return [_sanitize_for_json(item) for item in value]
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return float(value)
    return str(value)


def _coerce_balance_value(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value)
        if math.isnan(number) or math.isinf(number):
            return None
        return number
    parsed = _parse_currency(value)
    if parsed is None:
        return None
    if math.isnan(parsed) or math.isinf(parsed):
        return None
    return float(parsed)


def _coerce_account_index(value: Any, fallback: int) -> int:
    """Return a stable integer account index for merge-tag payloads."""

    if isinstance(value, bool):
        return fallback
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if stripped:
            try:
                return int(stripped)
            except ValueError:
                return fallback
    return fallback


def _build_raw_line_payload(raw_entry: Any) -> Optional[Dict[str, Any]]:
    if isinstance(raw_entry, Mapping):
        text = str(raw_entry.get("text") or "").strip()
        if not text:
            return None
        payload: Dict[str, Any] = {"text": text}
        for key in ("page", "line", "x0", "y0", "x1", "y1"):
            if key in raw_entry:
                payload[key] = raw_entry[key]
        return payload
    if isinstance(raw_entry, str):
        text = raw_entry.strip()
        if text:
            return {"text": text}
    return None


def _resolve_case_dir_for_account(
    sid: str,
    account_index: int,
    runs_root_path: Optional[Path],
    manifest: Optional[RunManifest],
    *,
    context_dir: Optional[Any] = None,
) -> Optional[Path]:
    if isinstance(context_dir, Path):
        return context_dir
    if isinstance(context_dir, str) and context_dir:
        return Path(context_dir)
    if runs_root_path is None or not sid:
        return None

    group = _manifest_group_key(account_index)
    base = _manifest_path(manifest, group, "dir")
    if base is not None:
        return base

    base = runs_root_path / sid / "cases" / "accounts" / str(account_index)
    if base.exists():
        return base
    alt_base = runs_root_path / "cases" / sid / "accounts" / str(account_index)
    if alt_base.exists():
        return alt_base
    return base


def _write_ai_pack(
    *,
    sid: str,
    runs_root_path: Optional[Path],
    manifest: Optional[RunManifest],
    idx_a: int,
    idx_b: int,
    score: float,
    parts: Mapping[str, float],
    aux: Mapping[str, Any],
    reasons: Optional[Mapping[str, Any]],
    account_contexts: List[Mapping[str, Any]],
) -> None:
    if runs_root_path is None or not sid:
        return
    if not isinstance(idx_a, int) or not isinstance(idx_b, int):
        return

    context_a: Mapping[str, Any] = {}
    context_b: Mapping[str, Any] = {}
    if 0 <= idx_a < len(account_contexts):
        context_a = account_contexts[idx_a]
    if 0 <= idx_b < len(account_contexts):
        context_b = account_contexts[idx_b]

    acc_idx_a = context_a.get("account_index")
    if not isinstance(acc_idx_a, int):
        acc_idx_a = idx_a

    case_dir = _resolve_case_dir_for_account(
        sid,
        acc_idx_a,
        runs_root_path,
        manifest,
        context_dir=context_a.get("case_dir"),
    )
    if case_dir is None:
        return

    ai_dir = case_dir / "ai"
    try:
        ai_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        logger.exception(
            "MERGE_SCORE sid=<%s> i=<%d> j=<%d> ai_pack_mkdir_failed path=%s",
            sid,
            idx_a,
            idx_b,
            ai_dir,
        )
        return

    pack_path = ai_dir / f"merge_pair_{idx_a}_{idx_b}.json"

    parts_payload: Dict[str, float] = {}
    for name, value in parts.items():
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isnan(number) or math.isinf(number):
            continue
        parts_payload[name] = number

    balance_payload = {
        "left": _coerce_balance_value(aux.get("balowed_a")),
        "right": _coerce_balance_value(aux.get("balowed_b")),
    }

    acct_a = aux.get("acct_num_a") if isinstance(aux, Mapping) else None
    acct_b = aux.get("acct_num_b") if isinstance(aux, Mapping) else None
    acct_payload = {
        "level": aux.get("acctnum_level") if isinstance(aux, Mapping) else None,
        "masked_any": bool(aux.get("acctnum_masked_any", False))
        if isinstance(aux, Mapping)
        else False,
        "left": _sanitize_for_json(acct_a) if acct_a is not None else None,
        "right": _sanitize_for_json(acct_b) if acct_b is not None else None,
    }
    if acct_payload["level"] is None:
        acct_payload["level"] = "none"

    sources: Dict[str, Any] = {}
    raw_line_context: Dict[str, Any] = {}

    left_source = context_a.get("acct_display_source") if isinstance(context_a, Mapping) else None
    right_source = context_b.get("acct_display_source") if isinstance(context_b, Mapping) else None
    if left_source:
        sources["left"] = left_source
    if right_source:
        sources["right"] = right_source

    if left_source == "raw_lines":
        payload = _build_raw_line_payload(context_a.get("acct_display_raw_line"))
        if payload:
            raw_line_context["left"] = payload
    if right_source == "raw_lines":
        payload = _build_raw_line_payload(context_b.get("acct_display_raw_line"))
        if payload:
            raw_line_context["right"] = payload

    display_map: Dict[str, Any] = {}
    left_display = context_a.get("account_number_display")
    right_display = context_b.get("account_number_display")
    if left_display not in (None, ""):
        display_map["left"] = left_display
    if right_display not in (None, ""):
        display_map["right"] = right_display

    pack: Dict[str, Any] = {
        "sid": sid,
        "decision": "ai",
        "pair": {"left_index": idx_a, "right_index": idx_b},
        "score": float(score),
        "parts": parts_payload,
        "balance_owed": balance_payload,
        "acctnum": acct_payload,
    }

    override_reasons = None
    if isinstance(aux, Mapping):
        override_reasons = aux.get("override_reasons")
    if override_reasons:
        pack["override_reasons"] = _sanitize_for_json(override_reasons)

    if sources:
        pack["acct_display_source"] = _sanitize_for_json(sources)
    if display_map:
        pack["account_number_display"] = _sanitize_for_json(display_map)
    if raw_line_context:
        pack["raw_lines"] = _sanitize_for_json(raw_line_context)

    if reasons:
        pack["reasons"] = _sanitize_for_json(reasons)

    sanitized_pack = _sanitize_for_json(pack)
    try:
        pack_text = json.dumps(sanitized_pack, ensure_ascii=False, indent=2, sort_keys=True)
        pack_path.write_text(pack_text, encoding="utf-8")
    except Exception:
        logger.exception(
            "MERGE_SCORE sid=<%s> i=<%d> j=<%d> ai_pack_write_failed path=%s",
            sid,
            idx_a,
            idx_b,
            pack_path,
        )

def cluster_problematic_accounts(
    candidates: List[Dict[str, Any]],
    cfg: LegacyMergeCfg = DEFAULT_CFG,
    *,
    sid: Optional[str] = None,
    runs_root: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Cluster problematic accounts using pairwise comparisons."""

    size = len(candidates)
    sid_value = sid or "-"
    runs_root_path: Optional[Path] = None
    if runs_root is not None:
        runs_root_path = Path(runs_root)
    elif sid_value != "-":
        env_root = os.getenv("RUNS_ROOT")
        runs_root_path = Path(env_root) if env_root else Path("runs")

    manifest: Optional[RunManifest] = None
    if runs_root_path is not None and sid_value != "-":
        manifest_path = runs_root_path / sid_value / "manifest.json"
        if manifest_path.exists():
            try:
                manifest = RunManifest(manifest_path).load()
            except Exception:
                logger.debug(
                    "MERGE_SCORE sid=<%s> manifest_load_failed path=%s",
                    sid_value,
                    manifest_path,
                )

    if size == 0:
        logger.info(
            "MERGE_SUMMARY sid=<%s> clusters=<0> auto_pairs=<0> ai_pairs=<0> skipped_pairs=<0>",
            sid_value,
        )
        return candidates

    prepared_accounts: List[Dict[str, Any]] = []
    account_contexts: List[Dict[str, Any]] = []
    for idx, candidate in enumerate(candidates):
        account_index = candidate.get("account_index")
        if not isinstance(account_index, int):
            account_index = idx

        prepared = dict(candidate)
        context: Dict[str, Any] = {"account_index": account_index}
        if sid_value != "-" and runs_root_path is not None:
            try:
                case_data = load_case_account(
                    sid_value, account_index, runs_root_path, manifest
                )
            except Exception:
                logger.debug(
                    "MERGE_SCORE sid=<%s> i=<%d> load_case_failed index=<%s>",
                    sid_value,
                    idx,
                    account_index,
                )
            else:
                prepared_case, prepared_meta = _prepare_account_for_scoring(case_data)
                merged_prepared = dict(candidate)
                merged_prepared.update(prepared_case)
                prepared = merged_prepared
                if isinstance(prepared_meta, Mapping):
                    context.update(dict(prepared_meta))
                context["case_dir"] = case_data.get("case_dir")
                context["raw_lines"] = case_data.get("raw_lines")
                for key in ("heading_guess", "account_id"):
                    if key in candidate and key not in prepared:
                        prepared[key] = candidate[key]

        prepared.setdefault("account_index", account_index)
        if "account_id" not in prepared and candidate.get("account_id") is not None:
            prepared["account_id"] = candidate["account_id"]
        prepared_accounts.append(prepared)
        account_contexts.append(context)

    pair_results: List[
        List[
            Optional[
                Tuple[float, str, Dict[str, float], Dict[str, Any], Dict[str, Any]]
            ]
        ]
    ] = [
        [None] * size for _ in range(size)
    ]
    auto_edges: List[Tuple[int, int]] = []
    ai_pairs: List[Tuple[int, int]] = []

    for i in range(size):
        for j in range(i + 1, size):
            score, parts, aux = score_accounts(prepared_accounts[i], prepared_accounts[j], cfg)
            base_score = score
            override_meta = {"sid": sid_value, "idx_a": i, "idx_b": j, "aux": aux}
            score, override_decision, reasons = _maybe_apply_balowed_override(
                parts, score, cfg, override_meta
            )
            if override_decision is None:
                decision = decide_merge(score, cfg)
            else:
                decision = override_decision
            score, decision, reasons, aux = _apply_account_number_ai_override(
                base_score,
                score,
                decision,
                aux,
                reasons,
                cfg.thresholds,
                sid_value=sid_value,
                idx_a=i,
                idx_b=j,
            )

            pair_results[i][j] = (score, decision, parts, aux, reasons)
            pair_results[j][i] = (score, decision, parts, aux, reasons)
            if decision == "auto":
                auto_edges.append((i, j))
            elif decision == "ai":
                ai_pairs.append((i, j))
                _write_ai_pack(
                    sid=sid_value,
                    runs_root_path=runs_root_path,
                    manifest=manifest,
                    idx_a=i,
                    idx_b=j,
                    score=score,
                    parts=parts,
                    aux=aux,
                    reasons=reasons,
                    account_contexts=account_contexts,
                )

            parts_log = json.dumps(
                {name: float(parts[name]) for name in sorted(parts.keys())},
                sort_keys=True,
            )
            logger.info(
                "MERGE_SCORE sid=<%s> i=<%d> j=<%d> parts=<%s> score=<%.4f>",
                sid_value,
                i,
                j,
                parts_log,
                score,
            )
            logger.info(
                "MERGE_DECISION sid=<%s> i=<%d> j=<%d> decision=<%s> score=<%.4f>",
                sid_value,
                i,
                j,
                decision,
                score,
            )

    graph = _build_auto_graph(size, auto_edges)
    components = _connected_components(graph)
    component_map: Dict[int, Tuple[int, List[int]]] = {}
    for idx, comp in enumerate(components, start=1):
        for node in comp:
            component_map[node] = (idx, comp)

    skip_pairs: Set[Tuple[int, int]] = set()
    for i in range(size):
        comp_i = component_map.get(i, (i + 1, [i]))
        comp_nodes_i = comp_i[1]
        if len(comp_nodes_i) <= 1:
            continue
        for j in range(i + 1, size):
            pair = pair_results[i][j]
            if pair is None:
                continue
            _, pair_decision, _, _, _ = pair
            if pair_decision == "auto":
                continue
            comp_j = component_map.get(j, (j + 1, [j]))
            if comp_i[0] == comp_j[0]:
                skip_pairs.add((i, j))

    ai_pair_keys = {(min(i, j), max(i, j)) for i, j in ai_pairs}
    skipped_keys = {(min(i, j), max(i, j)) for i, j in skip_pairs}
    effective_ai_pairs = sum(1 for key in ai_pair_keys if key not in skipped_keys)

    for i, account in enumerate(candidates):
        comp_idx, comp_nodes = component_map.get(i, (i + 1, [i]))
        group_id = f"g{comp_idx}"
        score_entries: List[Dict[str, Any]] = []
        best: Optional[Dict[str, Any]] = None
        best_parts: Dict[str, float] = {}
        best_aux: Dict[str, Any] = {}

        for j in range(size):
            if i == j:
                continue
            pair = pair_results[i][j]
            if pair is None:
                continue
            pair_key = (i, j) if i < j else (j, i)
            if pair_key in skip_pairs:
                continue
            pair_score, pair_decision, pair_parts, pair_aux, pair_reasons = pair
            other_account = prepared_accounts[j]
            other_index = _coerce_account_index(other_account.get("account_index"), j)
            entry = {
                "account_index": other_index,
                "score": pair_score,
                "decision": pair_decision,
            }
            if isinstance(other_account, Mapping):
                other_account_id = other_account.get("account_id")
                if other_account_id is not None:
                    entry["account_id"] = other_account_id
            if pair_reasons:
                reasons_payload = None
                if isinstance(pair_reasons, Mapping):
                    payload = pair_reasons.get("reasons")
                    if isinstance(payload, list):
                        reasons_payload = [dict(item) for item in payload]
                if reasons_payload is None and isinstance(pair_reasons, Mapping):
                    reasons_payload = dict(pair_reasons)
                if reasons_payload is not None:
                    entry["reasons"] = reasons_payload
            score_entries.append(entry)
            if best is None or pair_score > best["score"]:
                best = entry.copy()
                best_parts = pair_parts
                best_aux = pair_aux

        if len(comp_nodes) > 1:
            decision = "auto"
        elif best is not None:
            decision = best["decision"]
        else:
            decision = "different"

        merge_tag = {
            "group_id": group_id,
            "decision": decision,
            "score_to": sorted(score_entries, key=lambda item: item["score"], reverse=True),
            "best_match": best,
            "parts": best_parts,
        }
        if best is not None and best.get("reasons"):
            best_reasons = best["reasons"]
            if isinstance(best_reasons, list):
                merge_tag["reasons"] = [dict(item) for item in best_reasons]
            elif isinstance(best_reasons, Mapping):
                merge_tag["reasons"] = dict(best_reasons)
        elif best is not None and isinstance(best_aux, Mapping):
            aux_reasons = best_aux.get("override_reason_entries")
            if isinstance(aux_reasons, list) and aux_reasons:
                merge_tag["reasons"] = [dict(item) for item in aux_reasons]
        if best is not None:
            merge_tag["aux"] = best_aux
        account["merge_tag"] = merge_tag
        _persist_merge_tag(
            sid_value,
            account.get("account_index"),
            merge_tag,
            runs_root_path,
            manifest,
        )

    logger.info(
        "MERGE_SUMMARY sid=<%s> clusters=<%d> auto_pairs=<%d> ai_pairs=<%d> skipped_pairs=<%d>",
        sid_value,
        len(components),
        len(auto_edges),
        effective_ai_pairs,
        len(skip_pairs),
    )

    return candidates
