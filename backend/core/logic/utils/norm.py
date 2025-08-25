"""Heading normalization utilities."""

from __future__ import annotations

import re

from .names_normalization import COMMON_CREDITOR_ALIASES

_EXTRA_ALIASES = {
    "gs bank usa": "gs",
}

_ALIASES = {**COMMON_CREDITOR_ALIASES, **_EXTRA_ALIASES}

def normalize_heading(s: str) -> str:
    """Return a normalized account heading.

    The normalization is tolerant to punctuation, spacing, dashes and slashes
    and collapses common aliases to a canonical form.
    """
    if not s:
        return ""
    name = s.lower().strip()
    # Replace dashes and slashes with spaces then drop other punctuation
    name = re.sub(r"[/-]+", " ", name)
    name = re.sub(r"[^a-z0-9 ]", "", name)
    name = re.sub(r"\s+", " ", name)
    for alias, canonical in _ALIASES.items():
        if alias in name:
            return canonical
    name = re.sub(r"\b(bank|usa|na|n\.a\.|llc|inc|corp|co|company)\b", "", name)
    name = re.sub(r"\s+", " ", name)
    return name.strip()
