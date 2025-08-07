"""Utilities for analyzing custom client notes and related text."""
from __future__ import annotations

import re

from .names_normalization import normalize_creditor_name

HARDSHIP_RE = re.compile(
    r"(lost my job|job loss|layoff|medical|illness|hospital|covid|pandemic|family emergency|divorce|funeral|death in|financial hardship|hardship|sick)",
    re.I,
)


def is_general_hardship_note(text: str | None) -> bool:
    """Return True if the note appears to be a general hardship explanation."""
    if not text:
        return False
    return bool(HARDSHIP_RE.search(text.strip().lower()))


def analyze_custom_notes(custom_notes: dict, account_names: list[str]):
    """Separate account-specific notes from general hardship notes.

    Returns a tuple ``(specific_notes, general_notes)`` where ``specific_notes``
    is a mapping of normalized account names to notes.
    """

    normalized_accounts = {normalize_creditor_name(n) for n in account_names}
    specific: dict[str, str] = {}
    general: list[str] = []

    for key, note in (custom_notes or {}).items():
        if not note:
            continue
        key_norm = normalize_creditor_name(key)
        if key_norm in normalized_accounts and not is_general_hardship_note(note):
            specific[key_norm] = note.strip()
        else:
            general.append(str(note).strip())

    return specific, general


def get_client_address_lines(client_info: dict) -> list[str]:
    """Return client's mailing address lines.

    Priority order:
    1. ``client_info['address']``
    2. ``client_info['current_address']`` extracted from the credit report

    The returned list may contain one or two lines. When no address is found,
    an empty list is returned so the caller can render a placeholder line.
    """

    raw = (client_info.get("address") or client_info.get("current_address") or "").strip()
    if not raw:
        return []

    # Normalize separators to detect street vs city/state/zip parts
    raw = raw.replace("\r", "\n")
    parts = [p.strip() for p in re.split(r"\n|,", raw) if p.strip()]

    if len(parts) >= 2:
        line1 = parts[0]
        line2 = ", ".join(parts[1:])
        return [line1, line2]
    return [raw]
