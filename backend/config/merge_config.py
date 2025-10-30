"""Helpers for loading merge configuration from environment variables.

This module centralizes all parsing logic for MERGE_* environment variables so
that merge and deduplication behaviour can be controlled without code changes.
It ensures values such as booleans, numbers, and JSON payloads are converted to
native Python types and provides sensible defaults when variables are missing.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Any, Dict, List, Set

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

DEFAULT_CONFIG: Dict[str, Any] = {
    "enabled": False,
    "fields": list(DEFAULT_FIELDS),
    # Allowlist defaults mirror the historic field sequence so enforcement can
    # be toggled on without requiring explicit overrides.
    "fields_override": list(DEFAULT_FIELDS),
    "allowlist_enforce": False,
    # Custom weights are opt-in to preserve legacy scoring when disabled.
    "use_custom_weights": False,
    # Optional merge fields stay disabled until toggled via MERGE_USE_* flags.
    "use_original_creditor": False,
    "use_creditor_name": False,
    "weights": {},
    "thresholds": {},
    "overrides": {},
}


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


def _build_merge_config() -> Dict[str, Any]:
    """Construct the merge configuration from environment variables."""

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


@lru_cache(maxsize=1)
def get_merge_config() -> Dict[str, Any]:
    """Return cached merge configuration for reuse across the application."""

    # Cache ensures repeated calls are cheap while still reflecting the env state
    # from process startup. Using a helper makes it easy to reset in tests.
    return _build_merge_config()


def reset_merge_config_cache() -> None:
    """Clear cached merge configuration (primarily for testing support)."""

    # Providing a reset hook keeps behaviour explicit for unit tests.
    get_merge_config.cache_clear()

