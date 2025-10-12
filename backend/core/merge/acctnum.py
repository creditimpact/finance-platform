"""Account-number normalization and bureau-pair matching helpers."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Mapping

_MATCH_LEVEL = "exact_or_known_match"
_NONE_LEVEL = "none"

__all__ = [
    "AccountNumberMatch",
    "NormalizedAccountNumber",
    "acctnum_level",
    "acctnum_match_level",
    "acctnum_match_visible",
    "acctnum_visible_match",
    "best_account_number_match",
    "match_level",
    "normalize_display",
    "normalize_level",
]

_BUREAUS = ("transunion", "experian", "equifax")

_LEVEL_POINTS: Dict[str, int] = {
    _MATCH_LEVEL: 28,
}

_LEVEL_RANK: Dict[str, int] = {
    _NONE_LEVEL: 0,
    _MATCH_LEVEL: 1,
}

_MASK_CHARACTERS = frozenset({"*", "X", "x", "•", "#", "∙", "·", "●", "_"})
_ENDING_IN_PATTERN = re.compile(r"(?i)\bending[\s\-]*in\b")
_HYPHEN_CHARACTERS = frozenset("-‐‑‒–—―−")


def _derive_mask_metadata(raw: str, digits: str) -> tuple[str | None, bool, int]:
    """Return mask metadata derived from the raw display value."""

    if not raw:
        return None, False, 0

    cleaned = _ENDING_IN_PATTERN.sub(" ", raw)
    mask_found = False
    canonical_chars: list[str] = []

    for char in cleaned:
        if char in _MASK_CHARACTERS:
            canonical_chars.append("*")
            mask_found = True
        elif char.isdigit():
            canonical_chars.append(char)
        elif char.isspace() or char in _HYPHEN_CHARACTERS:
            continue
        else:
            canonical_chars.append(" ")

    if canonical_chars:
        canonical = "".join(canonical_chars)
        canonical = re.sub(r"\s+", " ", canonical).strip()
        canonical = canonical.replace(" ", "")
    else:
        canonical = ""

    canon_mask = canonical if mask_found and "*" in canonical else None

    visible_digits = len(digits) if digits else sum(ch.isdigit() for ch in raw)

    return canon_mask, mask_found, visible_digits


@dataclass(frozen=True)
class NormalizedAccountNumber:
    """Normalized representation of a bureau account number display."""

    raw: str
    digits: str

    @property
    def has_digits(self) -> bool:
        return bool(self.digits)

    @property
    def canon_mask(self) -> str | None:
        canon_mask, _, _ = _derive_mask_metadata(self.raw, self.digits)
        return canon_mask

    @property
    def has_mask(self) -> bool:
        _, has_mask, _ = _derive_mask_metadata(self.raw, self.digits)
        return has_mask

    @property
    def visible_digits(self) -> int:
        _, _, visible_digits = _derive_mask_metadata(self.raw, self.digits)
        return visible_digits

    def to_debug_dict(self) -> Dict[str, str]:
        return {
            "raw": self.raw,
            "digits": self.digits,
        }


_EMPTY_NORMALIZED = NormalizedAccountNumber("", "")


def normalize_display(display: str | None) -> NormalizedAccountNumber:
    """Normalize an ``account_number_display`` value to digits only."""

    raw = str(display or "")
    digits = re.sub(r"\D", "", raw)
    return NormalizedAccountNumber(raw, digits)


def normalize_level(level: str | None) -> str:
    """Clamp a free-form level value to the supported enumeration."""

    if isinstance(level, str):
        candidate = level.strip().lower()
        if candidate == _MATCH_LEVEL:
            return _MATCH_LEVEL
    return _NONE_LEVEL


@dataclass(frozen=True)
class AccountNumberMatch:
    """Best-match metadata between two normalized account numbers."""

    level: str
    a_bureau: str
    b_bureau: str
    a: NormalizedAccountNumber
    b: NormalizedAccountNumber
    debug: Dict[str, str]

    @property
    def points(self) -> int:
        return _LEVEL_POINTS.get(self.level, 0)

    def swapped(self) -> "AccountNumberMatch":
        return AccountNumberMatch(
            self.level,
            self.b_bureau,
            self.a_bureau,
            self.b,
            self.a,
            dict(self.debug),
        )


DIGITS = re.compile(r"\d")


def _digits(s: str) -> str:
    return "".join(DIGITS.findall(s or ""))


def _alnum(s: str) -> str:
    return "".join(ch for ch in (s or "") if ch.isalnum()).upper()


def _match_visible_digits(short: str, long_: str) -> tuple[bool, str]:
    """Return whether ``short`` appears sequentially inside ``long_``."""

    offset = long_.find(short)
    if offset == -1:
        return False, ""
    return True, str(offset)


def acctnum_visible_match(
    a_raw: str, b_raw: str
) -> tuple[bool, dict[str, dict[str, str] | str]]:
    a_digits = _digits(a_raw)
    b_digits = _digits(b_raw)

    debug: Dict[str, dict[str, str] | str] = {
        "a": {
            "raw": str(a_raw or ""),
            "digits": a_digits,
        },
        "b": {
            "raw": str(b_raw or ""),
            "digits": b_digits,
        },
        "short": "",
        "long": "",
        "why": "",
        "match_offset": "",
    }

    if not a_digits or not b_digits:
        debug["why"] = "missing_visible_digits"
        return False, debug

    if len(a_digits) <= len(b_digits):
        short, long_ = a_digits, b_digits
    else:
        short, long_ = b_digits, a_digits

    debug["short"] = short
    debug["long"] = long_

    ok, offset = _match_visible_digits(short, long_)
    if ok:
        debug["match_offset"] = offset
        return True, debug

    debug["why"] = "visible_digits_conflict"
    return False, debug


def acctnum_match_visible(
    a_raw: str, b_raw: str
) -> tuple[bool, dict[str, dict[str, str] | str]]:
    """Compatibility wrapper for legacy call sites."""

    return acctnum_visible_match(a_raw, b_raw)


def acctnum_match_level(
    a_raw: str, b_raw: str
) -> tuple[str, dict[str, dict[str, str] | str]]:
    ok, dbg = acctnum_visible_match(a_raw, b_raw)
    return ("exact_or_known_match" if ok else "none"), dbg


def acctnum_level(a_raw: str, b_raw: str) -> tuple[str, Dict[str, dict[str, str] | str]]:
    """Return the account-number level and debug metadata."""

    a_raw_str = str(a_raw or "")
    b_raw_str = str(b_raw or "")

    a_digits = _digits(a_raw_str)
    b_digits = _digits(b_raw_str)
    a_alnum = _alnum(a_raw_str)
    b_alnum = _alnum(b_raw_str)

    debug: Dict[str, dict[str, str] | str] = {
        "a": {"raw": a_raw_str, "digits": a_digits, "alnum": a_alnum},
        "b": {"raw": b_raw_str, "digits": b_digits, "alnum": b_alnum},
        "short": "",
        "long": "",
        "why": "",
        "mode": "none",
        "match_offset": "",
    }

    if a_digits and b_digits:
        if len(a_digits) <= len(b_digits):
            short, long_ = a_digits, b_digits
        else:
            short, long_ = b_digits, a_digits
        debug["short"] = short
        debug["long"] = long_
        ok, offset = _match_visible_digits(short, long_)
        if ok:
            debug["match_offset"] = offset

    if a_alnum and b_alnum and a_alnum == b_alnum:
        debug["mode"] = "alnum"
        debug["why"] = "alnum_match"
        return _MATCH_LEVEL, debug

    if (
        a_digits
        and b_digits
        and a_digits == b_digits
        and a_alnum.isdigit()
        and b_alnum.isdigit()
    ):
        debug["mode"] = "digits"
        debug["why"] = "digits_match"
        return _MATCH_LEVEL, debug

    if not a_digits and not b_digits:
        debug["why"] = "empty"
    elif (
        a_alnum
        and b_alnum
        and a_alnum != b_alnum
        and (not a_alnum.isdigit() or not b_alnum.isdigit())
    ):
        debug["why"] = "alnum_conflict"
    elif a_digits and b_digits and a_digits != b_digits:
        debug["why"] = "digit_conflict"
    elif a_alnum and b_alnum and a_alnum != b_alnum:
        debug["why"] = "alnum_conflict"
    else:
        debug["why"] = "insufficient_data"

    return _NONE_LEVEL, debug


def match_level(a: NormalizedAccountNumber, b: NormalizedAccountNumber) -> str:
    """Return the strict account-number level between two normalized numbers."""

    level, _ = acctnum_level(a.raw, b.raw)
    return level


def best_account_number_match(
    a_map: Mapping[str, NormalizedAccountNumber],
    b_map: Mapping[str, NormalizedAccountNumber],
) -> AccountNumberMatch:
    """Compute the best bureau pairing by strict account-number level."""

    best_match = AccountNumberMatch(
        _NONE_LEVEL,
        "",
        "",
        _EMPTY_NORMALIZED,
        _EMPTY_NORMALIZED,
        {"short": "", "long": ""},
    )
    best_rank = -1

    for a_bureau in _BUREAUS:
        a_norm = a_map.get(a_bureau, _EMPTY_NORMALIZED)
        for b_bureau in _BUREAUS:
            b_norm = b_map.get(b_bureau, _EMPTY_NORMALIZED)
            level, debug = acctnum_level(a_norm.raw, b_norm.raw)
            rank = _LEVEL_RANK[level]
            if rank > best_rank:
                best_rank = rank
                best_match = AccountNumberMatch(
                    level,
                    a_bureau,
                    b_bureau,
                    a_norm,
                    b_norm,
                    debug,
                )
            if rank == _LEVEL_RANK["exact_or_known_match"]:
                break
        if best_rank == _LEVEL_RANK["exact_or_known_match"]:
            break

    return best_match
