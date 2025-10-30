"""Helpers for loading merge configuration from environment variables.

This module centralizes all parsing logic for ``MERGE_*`` environment variables
so that merge and deduplication behaviour can be controlled without code
changes. It ensures values such as booleans, numbers, and JSON payloads are
converted to native Python types and provides sensible defaults when variables
are missing.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from collections.abc import Mapping
from typing import Any, Dict, Iterable, Iterator, List, Optional, Set

import logging

# Base prefix for all merge related environment variables.
MERGE_PREFIX = "MERGE_"

# Default configuration keeps backward compatible behaviour when flags are not
# provided via environment variables.
DEFAULT_FIELDS: List[str] = [
    "account_number",
    "date_opened",
    "balance_owed",
    "account_type",
    "account_status",
    "history_2y",
    "history_7y",
]

logger = logging.getLogger(__name__)


POINTS_MODE_DEFAULT_WEIGHTS: Dict[str, float] = {
    "account_number": 1.0,
    "date_opened": 1.0,
    "balance_owed": 3.0,
    "account_type": 0.5,
    "account_status": 0.5,
    "history_2y": 1.0,
    "history_7y": 1.0,
}


DEFAULT_CONFIG: Dict[str, Any] = {
    "enabled": False,
    "fields": list(DEFAULT_FIELDS),
    # Allowlist defaults mirror the historic field sequence so enforcement can
    # be toggled on without requiring explicit overrides.
    "fields_override": list(DEFAULT_FIELDS),
    "allowlist_enforce": False,
    # Debug logging remains opt-in so production noise does not increase.
    "debug": False,
    # ``log_every`` controls the sampling cadence for debug logs when enabled.
    "log_every": 0,
    # Custom weights are opt-in to preserve legacy scoring when disabled.
    "use_custom_weights": False,
    # Optional merge fields stay disabled until toggled via MERGE_USE_* flags.
    "use_original_creditor": False,
    "use_creditor_name": False,
    "weights": {},
    "thresholds": {},
    "overrides": {},
    # Points-based scoring is disabled by default so legacy behaviour remains
    # active unless the new flag is explicitly enabled.
    "points_mode": False,
    "ai_points_threshold": 3.0,
    "direct_points_threshold": 5.0,
    # Emit per-pair diagnostics for the first N comparisons when points mode
    # is active so behaviour can be audited without enabling verbose logging.
    "points_diagnostics_limit": 3,
}

ALLOWLIST_FIELDS: List[str] = list(DEFAULT_FIELDS)

_MERGE_CONFIG_LOGGED = False


class MergeConfig(Mapping[str, Any]):
    """Mapping-compatible wrapper exposing structured merge configuration."""

    def __init__(
        self,
        raw: Dict[str, Any],
        *,
        points_mode: bool,
        ai_points_threshold: float,
        direct_points_threshold: float,
        allowlist_enforce: bool,
        fields: List[str],
        weights: Dict[str, float],
        tolerances: Dict[str, Any],
        points_diagnostics_limit: int,
    ) -> None:
        self._raw: Dict[str, Any] = dict(raw)
        self.points_mode = points_mode
        self.ai_points_threshold = ai_points_threshold
        self.direct_points_threshold = direct_points_threshold
        self.allowlist_enforce = allowlist_enforce
        self.fields = list(fields)
        self.weights = dict(weights)
        self.tolerances = dict(tolerances)
        self.points_diagnostics_limit = int(max(points_diagnostics_limit, 0))

        # Surface structured values through the mapping interface for backward
        # compatibility with existing dictionary-based access patterns.
        self._raw.update(
            {
                "points_mode": self.points_mode,
                "ai_points_threshold": self.ai_points_threshold,
                "direct_points_threshold": self.direct_points_threshold,
                "allowlist_enforce": self.allowlist_enforce,
                "fields": list(self.fields),
                "weights": dict(self.weights),
                "tolerances": dict(self.tolerances),
                "points_diagnostics_limit": self.points_diagnostics_limit,
            }
        )

    def __getitem__(self, key: str) -> Any:
        return self._raw[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._raw)

    def __len__(self) -> int:
        return len(self._raw)

    def get(self, key: str, default: Any = None) -> Any:
        return self._raw.get(key, default)

    def keys(self) -> Iterable[str]:
        return self._raw.keys()

    def items(self):
        return self._raw.items()

    def values(self):
        return self._raw.values()



def _parse_json(value: str) -> Any:
    """Parse JSON value while gracefully handling malformed payloads."""

    try:
        # We guard json loads so that a malformed value does not crash startup.
        return json.loads(value)
    except json.JSONDecodeError:
        # Fall back to raw string if parsing fails to preserve backward compat.
        return value


def _parse_env_value(env_key: str, raw_value: str) -> Any:
    """Convert a raw environment string into an appropriate Python type."""

    value = raw_value.strip()

    # Automatically decode *_JSON variables first to support structured config.
    if env_key.endswith("_JSON"):
        parsed = _parse_json(value)
        return parsed

    lowered = value.lower()
    if lowered in {"true", "false"}:
        # Translate human-friendly boolean strings into actual booleans.
        return lowered == "true"

    # Attempt integer parsing before floats to retain whole number semantics.
    if lowered.isdigit() or (lowered.startswith("-") and lowered[1:].isdigit()):
        try:
            return int(lowered)
        except ValueError:
            pass

    # Support floats that include decimal points or scientific notation.
    try:
        if any(char in lowered for char in [".", "e"]):
            return float(value)
    except ValueError:
        pass

    # Allow comma separated strings to represent field lists when JSON is not used.
    if "," in value:
        return [item.strip() for item in value.split(",") if item.strip()]

    # Fallback: retain original string for unhandled cases.
    return raw_value


def _coerce_bool(value: Any, default: bool = False) -> bool:
    """Convert arbitrary inputs into a boolean with a sensible default."""

    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "t", "yes", "y", "on"}:
            return True
        if lowered in {"0", "false", "f", "no", "n", "off"}:
            return False
    return default


def _coerce_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    """Convert input to float when possible, otherwise return the default."""

    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return default
    return default


def _coerce_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    """Convert input to integer when possible, otherwise return the default."""

    if value is None:
        return default
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        try:
            return int(float(value.strip()))
        except ValueError:
            return default
    return default


def _normalize_fields(value: Any) -> List[str]:
    """Normalize any field override input into a clean list of strings."""

    if value is None:
        return []
    if isinstance(value, str):
        value = [segment.strip() for segment in value.split(",") if segment.strip()]
    elif isinstance(value, (set, tuple)):
        value = list(value)
    if isinstance(value, list):
        seen: Set[str] = set()
        normalized: List[str] = []
        for item in value:
            if item is None:
                continue
            field = str(item).strip()
            if not field or field in seen:
                continue
            seen.add(field)
            normalized.append(field)
        return normalized
    return []


def _normalize_weights(
    raw_weights: Any, *, allowlist_enforce: bool, fields: List[str]
) -> Dict[str, float]:
    """Normalize configured weights into a float mapping keyed by field name."""

    weights: Dict[str, float] = {}
    if isinstance(raw_weights, Mapping):
        for key, value in raw_weights.items():
            field = str(key).strip()
            if not field:
                continue
            if allowlist_enforce and field not in ALLOWLIST_FIELDS:
                continue
            coerced = _coerce_float(value)
            if coerced is None:
                continue
            weights[field] = coerced

    resolved: Dict[str, float] = {}
    for field in fields:
        weight = weights.get(field)
        if weight is None:
            resolved[field] = 1.0
        else:
            resolved[field] = float(weight)

    return resolved


def _resolve_fields(
    *,
    allowlist_enforce: bool,
    override: List[str],
    configured_fields: List[str],
) -> List[str]:
    """Resolve the effective field list respecting allowlist enforcement."""

    allowed = set(ALLOWLIST_FIELDS)
    source_fields = override if override else configured_fields
    if not source_fields:
        source_fields = list(ALLOWLIST_FIELDS)

    resolved = [field for field in source_fields if field in allowed]

    if not allowlist_enforce:
        for field in ALLOWLIST_FIELDS:
            if field not in resolved:
                resolved.append(field)

    return resolved


def _resolve_tolerances(config: Dict[str, Any]) -> Dict[str, Any]:
    """Collect tolerance-related values from the raw environment configuration."""

    source = config.get("tolerances")
    if not isinstance(source, Mapping):
        source = {}

    def _lookup(key: str, fallback_key: str) -> Any:
        if key in source:
            return source.get(key)
        return config.get(fallback_key)

    return {
        "MERGE_TOL_DATE_DAYS": _coerce_int(
            _lookup("MERGE_TOL_DATE_DAYS", "tol_date_days")
        ),
        "MERGE_TOL_BALANCE_ABS": _coerce_float(
            _lookup("MERGE_TOL_BALANCE_ABS", "tol_balance_abs")
        ),
        "MERGE_TOL_BALANCE_RATIO": _coerce_float(
            _lookup("MERGE_TOL_BALANCE_RATIO", "tol_balance_ratio")
        ),
        "MERGE_ACCOUNTNUMBER_MATCH_MINLEN": _coerce_int(
            _lookup("MERGE_ACCOUNTNUMBER_MATCH_MINLEN", "accountnumber_match_minlen")
        ),
        "MERGE_HISTORY_SIMILARITY_THRESHOLD": _coerce_float(
            _lookup("MERGE_HISTORY_SIMILARITY_THRESHOLD", "history_similarity_threshold")
        ),
    }


def _build_merge_config() -> Dict[str, Any]:
    """Construct the merge configuration from the dedicated ``MERGE_*`` block."""

    config: Dict[str, Any] = dict(DEFAULT_CONFIG)
    present_keys: Set[str] = set()

    for key, raw_value in os.environ.items():
        if not key.startswith(MERGE_PREFIX):
            continue

        parsed_value = _parse_env_value(key, raw_value)
        short_key = key[len(MERGE_PREFIX) :].lower()

        # Normalize *_json keys to expose cleaner dictionary names (e.g., weights).
        if short_key.endswith("_json"):
            short_key = short_key[:-5]

        # Update the runtime configuration using the normalized key.
        config[short_key] = parsed_value
        # Track which keys were explicitly provided so that callers can
        # distinguish default values from environment overrides.
        present_keys.add(short_key)

    config["_present_keys"] = frozenset(present_keys)

    return config


def _create_structured_config(raw_config: Dict[str, Any]) -> MergeConfig:
    """Produce the structured ``MergeConfig`` wrapper from raw environment data."""

    allowlist_enforce = _coerce_bool(raw_config.get("allowlist_enforce"))
    points_mode = _coerce_bool(raw_config.get("points_mode"))
    ai_points_threshold = _coerce_float(raw_config.get("ai_points_threshold"), 3.0) or 3.0
    direct_points_threshold = (
        _coerce_float(raw_config.get("direct_points_threshold"), 5.0) or 5.0
    )
    diagnostics_limit = _coerce_int(
        raw_config.get("points_diagnostics_limit"),
        int(DEFAULT_CONFIG["points_diagnostics_limit"]),
    )
    if diagnostics_limit is None:
        diagnostics_limit = int(DEFAULT_CONFIG["points_diagnostics_limit"])
    diagnostics_limit = max(int(diagnostics_limit), 0)

    fields_override = _normalize_fields(raw_config.get("fields_override"))
    configured_fields = _normalize_fields(raw_config.get("fields"))
    effective_allowlist_enforce = allowlist_enforce or points_mode

    fields = _resolve_fields(
        allowlist_enforce=effective_allowlist_enforce,
        override=fields_override,
        configured_fields=configured_fields,
    )

    use_custom_weights = _coerce_bool(raw_config.get("use_custom_weights"))
    weights_source: Any = {}
    if points_mode:
        weights_source = raw_config.get("weights")
        if not isinstance(weights_source, Mapping) or not weights_source:
            weights_source = dict(POINTS_MODE_DEFAULT_WEIGHTS)
    elif use_custom_weights:
        weights_source = raw_config.get("weights")
    weights = _normalize_weights(
        weights_source,
        allowlist_enforce=effective_allowlist_enforce,
        fields=fields,
    )

    tolerances = _resolve_tolerances(raw_config)

    processed_raw = dict(raw_config)
    processed_raw["use_custom_weights"] = bool(
        raw_config.get("use_custom_weights")
    ) or bool(points_mode)
    processed_raw["fields_override"] = fields_override

    global _MERGE_CONFIG_LOGGED
    if not _MERGE_CONFIG_LOGGED:
        message = (
            "[MERGE] Config points_mode=%s ai_threshold=%.2f direct_threshold=%.2f "
            "fields=%s weights=%s tolerances=%s diagnostics_limit=%s"
            % (
                points_mode,
                float(ai_points_threshold),
                float(direct_points_threshold),
                fields,
                weights,
                tolerances,
                diagnostics_limit,
            )
        )
        logger.info(message)
        print(message)
        _MERGE_CONFIG_LOGGED = True

    return MergeConfig(
        processed_raw,
        points_mode=points_mode,
        ai_points_threshold=float(ai_points_threshold),
        direct_points_threshold=float(direct_points_threshold),
        allowlist_enforce=allowlist_enforce,
        fields=fields,
        weights=weights,
        tolerances=tolerances,
        points_diagnostics_limit=diagnostics_limit,
    )


@lru_cache(maxsize=1)
def get_merge_config() -> MergeConfig:
    """Return cached merge configuration for reuse across the application."""

    # Cache ensures repeated calls are cheap while still reflecting the env state
    # from process startup. Using a helper makes it easy to reset in tests.
    return _create_structured_config(_build_merge_config())


def reset_merge_config_cache() -> None:
    """Clear cached merge configuration (primarily for testing support)."""

    # Providing a reset hook keeps behaviour explicit for unit tests.
    get_merge_config.cache_clear()

