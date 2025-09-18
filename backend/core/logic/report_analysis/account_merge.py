"""Tools for scoring and clustering problematic accounts prior to merging."""

from __future__ import annotations

import difflib
import logging
import re
import os
from dataclasses import dataclass
from datetime import date
from typing import Any, Dict, Iterable, List, Mapping, Optional, Set, Tuple


logger = logging.getLogger(__name__)


@dataclass
class MergeCfg:
    """Configuration knobs for the merge pipeline."""

    weights: Dict[str, float]
    thresholds: Dict[str, float]
    acctnum_trigger_ai: str = "any"
    acctnum_min_score: float = 0.31
    acctnum_require_masked: bool = False
    # optional knobs: date_tolerance_days, balance_tolerance_ratio, etc.


_DEFAULT_THRESHOLDS = {
    "auto_merge_min": 0.78,
    "ai_band_min": 0.35,
    "ai_band_max": 0.78,
    "ai_hard_min": 0.30,
}

_DEFAULT_WEIGHTS = {
    "acct_num": 0.30,
    "dates": 0.20,
    "balance": 0.20,
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
    "acct_num": "MERGE_W_ACCT",
    "dates": "MERGE_W_DATES",
    "balance": "MERGE_W_BAL",
    "status": "MERGE_W_STATUS",
    "strings": "MERGE_W_STR",
}

_ACCTNUM_TRIGGER_CHOICES = {"off", "exact", "last4", "any"}
_ACCTNUM_TRIGGER_KEY = "MERGE_ACCTNUM_TRIGGER_AI"
_ACCTNUM_MIN_SCORE_KEY = "MERGE_ACCTNUM_MIN_SCORE"
_ACCTNUM_REQUIRE_MASKED_KEY = "MERGE_ACCTNUM_REQUIRE_MASKED"

_ACCT_NUMBER_FIELDS = ("account_number", "acct_num", "number")
_MASKED_ACCOUNT_PATTERN = re.compile(r"[xX*#•]")


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


def load_config_from_env(env: Optional[Mapping[str, str]] = None) -> MergeCfg:
    """Create a MergeCfg using optional environment overrides."""

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

    acctnum_trigger_ai = _read_env_choice(
        env_mapping, _ACCTNUM_TRIGGER_KEY, _DEFAULT_ACCTNUM_TRIGGER, _ACCTNUM_TRIGGER_CHOICES
    )
    acctnum_min_score = _read_env_float(
        env_mapping, _ACCTNUM_MIN_SCORE_KEY, _DEFAULT_ACCTNUM_MIN_SCORE
    )
    acctnum_require_masked = _read_env_flag(
        env_mapping, _ACCTNUM_REQUIRE_MASKED_KEY, _DEFAULT_ACCTNUM_REQUIRE_MASKED
    )

    return MergeCfg(
        weights=weights,
        thresholds=thresholds,
        acctnum_trigger_ai=acctnum_trigger_ai,
        acctnum_min_score=acctnum_min_score,
        acctnum_require_masked=acctnum_require_masked,
    )


DEFAULT_CFG = load_config_from_env()

_DATE_FIELDS = ("date_opened", "date_of_last_activity", "closed_date")
_BALANCE_FIELDS = ("past_due_amount", "balance_owed")
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
    return {
        "acct_num_raw": raw,
        "acct_num_digits": digits,
        "acct_num_masked": masked,
        "acct_num_last4": last4,
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


def _score_numeric(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None:
        return None
    max_val = max(abs(a), abs(b), 1.0)
    delta = abs(a - b)
    return max(0.0, 1.0 - min(delta / max_val, 1.0))


def _score_balance(acc_a: Dict[str, Any], acc_b: Dict[str, Any]) -> float:
    scores: List[float] = []
    parsed_a = {field: _parse_currency(acc_a.get(field)) for field in _BALANCE_FIELDS}
    parsed_b = {field: _parse_currency(acc_b.get(field)) for field in _BALANCE_FIELDS}

    if parsed_a.get("past_due_amount") is not None and parsed_b.get("past_due_amount") is not None:
        val = _score_numeric(parsed_a["past_due_amount"], parsed_b["past_due_amount"])
        if val is not None:
            scores.append(val)

    if parsed_a.get("balance_owed") is not None and parsed_b.get("balance_owed") is not None:
        val = _score_numeric(parsed_a["balance_owed"], parsed_b["balance_owed"])
        if val is not None:
            scores.append(val)

    if not scores:
        return 0.0
    return sum(scores) / len(scores)


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
    accA: Dict[str, Any], accB: Dict[str, Any], cfg: MergeCfg = DEFAULT_CFG
) -> Tuple[float, Dict[str, float], Dict[str, Any]]:
    """Return overall score, per-part contributions, and auxiliary details."""

    acct_score, acct_aux = _score_account_number(accA, accB)

    parts = {
        "acct_num": acct_score,
        "dates": _score_dates(accA, accB),
        "balance": _score_balance(accA, accB),
        "status": _score_status(accA, accB),
        "strings": _score_strings(accA, accB),
    }

    total_weight = sum(cfg.weights.get(name, 0.0) for name in parts)
    if total_weight <= 0:
        return 0.0, parts, acct_aux

    weighted = sum(parts[name] * cfg.weights.get(name, 0.0) for name in parts)
    score = weighted / total_weight
    return score, parts, acct_aux


def decide_merge(score: float, cfg: MergeCfg = DEFAULT_CFG) -> str:
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

    level = aux.get("acctnum_level")
    masked_any = bool(aux.get("acctnum_masked_any", False))

    env = os.environ
    trig = _read_env_choice(
        env, _ACCTNUM_TRIGGER_KEY, _DEFAULT_ACCTNUM_TRIGGER, _ACCTNUM_TRIGGER_CHOICES
    )
    minscore = _read_env_float(env, _ACCTNUM_MIN_SCORE_KEY, _DEFAULT_ACCTNUM_MIN_SCORE)
    req_masked = _read_env_flag(
        env, _ACCTNUM_REQUIRE_MASKED_KEY, _DEFAULT_ACCTNUM_REQUIRE_MASKED
    )

    eligible = (
        (trig == "any" and level in {"exact", "last4"})
        or (trig == "exact" and level == "exact")
        or (trig == "last4" and level == "last4")
    )
    if req_masked:
        eligible = eligible and masked_any

    if eligible and base_score < ai_band_min:
        lifted_score = max(score, minscore, ai_hard_min)
        updated_reasons = dict(reasons)
        updated_reasons["acctnum_only_triggers_ai"] = True
        updated_reasons["acctnum_match_level"] = level
        updated_reasons["acctnum_masked_any"] = masked_any

        aux_with_reasons = dict(aux)
        aux_with_reasons["override_reasons"] = dict(updated_reasons)

        logger.info(
            "MERGE_OVERRIDE sid=<%s> i=<%d> j=<%d> reason=acctnum_only_triggers_ai "
            "level=<%s> masked_any=<%d> lifted_to=<%.4f>",
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
    return score, decision, reasons, aux


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


def cluster_problematic_accounts(
    candidates: List[Dict[str, Any]],
    cfg: MergeCfg = DEFAULT_CFG,
    *,
    sid: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Cluster problematic accounts using pairwise comparisons."""

    size = len(candidates)
    sid_value = sid or "-"
    if size == 0:
        logger.info(
            "MERGE_SUMMARY  sid=<%s> clusters=<0> auto_pairs=<0> ai_pairs=<0> skipped_pairs=<0>",
            sid_value,
        )
        return candidates

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
            score, parts, aux = score_accounts(candidates[i], candidates[j], cfg)
            base_score = score
            decision = decide_merge(score, cfg)
            reasons: Dict[str, Any] = {}
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

            parts_str = ",".join(
                f"{name}:{parts[name]:.4f}" for name in sorted(parts.keys())
            )
            logger.info(
                "MERGE_DECISION sid=<%s> accA=<%d> accB=<%d> score=<%.4f> decision=<%s> parts=<%s>",
                sid_value,
                i,
                j,
                score,
                decision,
                parts_str,
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
            entry = {
                "account_index": j,
                "score": pair_score,
                "decision": pair_decision,
            }
            if pair_reasons:
                entry["reasons"] = dict(pair_reasons)
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
        if best is not None:
            merge_tag["aux"] = best_aux
        account["merge_tag"] = merge_tag

    logger.info(
        "MERGE_SUMMARY  sid=<%s> clusters=<%d> auto_pairs=<%d> ai_pairs=<%d> skipped_pairs=<%d>",
        sid_value,
        len(components),
        len(auto_edges),
        effective_ai_pairs,
        len(skip_pairs),
    )

    return candidates
