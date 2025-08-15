"""Helpers for redacting personally identifiable information from text.

This module provides simple pattern based masking for common PII. Emails and
phone numbers are fully redacted, while SSNs and account numbers keep their
last four digits for audit/debugging purposes.
"""

from __future__ import annotations

import re

_EMAIL_RE = re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")
_PHONE_RE = re.compile(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b")
_SSN_RE = re.compile(r"\b(?:\d{3}[- ]\d{2}[- ]\d{4}|\d{9})\b")
_ACCOUNT_RE = re.compile(r"\b(?:\d[ -]?){11,15}\d\b")


def redact_pii(text: str) -> str:
    """Return ``text`` with common PII patterns masked."""
    if not text:
        return ""
    redacted = _EMAIL_RE.sub("[REDACTED]", text)
    redacted = _PHONE_RE.sub("[REDACTED]", redacted)
    redacted = _SSN_RE.sub(
        lambda m: "***-**-" + re.sub(r"\D", "", m.group())[-4:], redacted
    )
    redacted = _ACCOUNT_RE.sub(
        lambda m: "****" + re.sub(r"\D", "", m.group())[-4:], redacted
    )
    return redacted
