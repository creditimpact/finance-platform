"""Utilities for working with date conventions in validation."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Mapping, Optional

from .config import get_prevalidation_trace_relpath

_LOGGER = logging.getLogger(__name__)

_VALID_CONVENTIONS = {"DMY", "MDY", "YMD"}


def _normalize_convention(value: Any) -> Optional[str]:
    """Return a normalised convention value if valid, otherwise ``None``."""

    if isinstance(value, str):
        candidate = value.strip().upper()
        if candidate in _VALID_CONVENTIONS:
            return candidate
    return None


def _extract_convention(payload: Any) -> Optional[str]:
    """Extract the convention value from a parsed JSON payload."""

    if not isinstance(payload, Mapping):
        return None

    for key in ("convention", "conv"):
        if key in payload:
            convention = _normalize_convention(payload[key])
            if convention is not None:
                return convention
    return None


def load_date_convention_for_sid(runs_root: str, sid: str, rel_path: str | None = None) -> str:
    """Load the date parsing convention for a given run.

    Parameters
    ----------
    runs_root:
        Absolute path to the directory containing run folders.
    sid:
        Identifier for the specific run whose convention we want to read.
    rel_path:
        Optional relative path to the convention file. When ``None`` the
        ``PREVALIDATION_OUT_PATH_REL`` environment variable (via
        :func:`get_prevalidation_trace_relpath`) determines the location.

    Returns
    -------
    str
        The date convention value (``"DMY"``, ``"MDY"`` or ``"YMD"``). If the
        file is missing or contains an invalid payload the default ``"MDY"`` is
        returned.
    """

    resolved_rel_path = rel_path or get_prevalidation_trace_relpath()
    convention_path = Path(runs_root) / sid / resolved_rel_path

    try:
        data = json.loads(convention_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        _LOGGER.debug("Date convention file not found at %s", convention_path)
        return "MDY"
    except (OSError, json.JSONDecodeError) as exc:
        _LOGGER.debug("Failed to read date convention file at %s: %s", convention_path, exc)
        return "MDY"

    convention = _extract_convention(data)
    if convention is None:
        _LOGGER.debug("Date convention not present or invalid in %s", convention_path)
        return "MDY"

    return convention


__all__ = ["load_date_convention_for_sid"]
