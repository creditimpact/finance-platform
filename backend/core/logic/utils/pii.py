"""Helpers for redacting personally identifiable information from text."""

from __future__ import annotations

import re

_EMAIL_RE = re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")
_PHONE_RE = re.compile(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b")
_SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_ACCOUNT_RE = re.compile(r"\b\d{12,16}\b")

_PATTERNS = (_EMAIL_RE, _PHONE_RE, _SSN_RE, _ACCOUNT_RE)


def redact_pii(text: str) -> str:
    """Return ``text`` with common PII patterns replaced by ``[REDACTED]``."""
    if not text:
        return ""
    redacted = text
    for pattern in _PATTERNS:
        redacted = pattern.sub("[REDACTED]", redacted)
    return redacted
