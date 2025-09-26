"""Account-number normalization and bureau-pair matching helpers."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Mapping

__all__ = [
    "AccountNumberMatch",
    "NormalizedAccountNumber",
    "best_account_number_match",
    "match_level",
    "normalize_display",
]

_BUREAUS = ("transunion", "experian", "equifax")

_LEVEL_POINTS: Dict[str, int] = {
    "exact": 40,
    "last6_bin": 32,
    "last6": 28,
}

_LEVEL_RANK: Dict[str, int] = {
    "none": 0,
    "last6": 1,
    "last6_bin": 2,
    "exact": 3,
}


@dataclass(frozen=True)
class NormalizedAccountNumber:
    """Normalized representation of a bureau account number display."""

    raw: str
    digits: str
    digits_last4: str
    digits_last5: str
    digits_last6: str
    digits_first6: str

    @property
    def has_digits(self) -> bool:
        return bool(self.digits)

    def to_debug_dict(self) -> Dict[str, str]:
        return {
            "raw": self.raw,
            "digits": self.digits,
            "digits_last4": self.digits_last4,
            "digits_last5": self.digits_last5,
            "digits_last6": self.digits_last6,
            "digits_first6": self.digits_first6,
        }


_EMPTY_NORMALIZED = NormalizedAccountNumber("", "", "", "", "", "")


def normalize_display(display: str | None) -> NormalizedAccountNumber:
    """Normalize an ``account_number_display`` value to digits and helpers."""

    raw = str(display or "")
    digits = re.sub(r"\D", "", raw)
    last4 = digits[-4:] if len(digits) >= 4 else ""
    last5 = digits[-5:] if len(digits) >= 5 else ""
    last6 = digits[-6:] if len(digits) >= 6 else ""
    bin6 = digits[:6] if len(digits) >= 6 else ""
    return NormalizedAccountNumber(raw, digits, last4, last5, last6, bin6)


@dataclass(frozen=True)
class AccountNumberMatch:
    """Best-match metadata between two normalized account numbers."""

    level: str
    a_bureau: str
    b_bureau: str
    a: NormalizedAccountNumber
    b: NormalizedAccountNumber

    @property
    def points(self) -> int:
        return _LEVEL_POINTS.get(self.level, 0)

    def swapped(self) -> "AccountNumberMatch":
        return AccountNumberMatch(self.level, self.b_bureau, self.a_bureau, self.b, self.a)


def match_level(a: NormalizedAccountNumber, b: NormalizedAccountNumber) -> str:
    """Return the strict account-number level between two normalized numbers."""

    if not (a.has_digits and b.has_digits):
        return "none"

    if len(a.digits) >= 8 and a.digits == b.digits:
        return "exact"

    if a.digits_last6 and b.digits_last6:
        if a.digits_first6 and a.digits_first6 == b.digits_first6 and a.digits_last6 == b.digits_last6:
            return "last6_bin"
        if a.digits_last6 == b.digits_last6:
            return "last6"

    return "none"


def best_account_number_match(
    a_map: Mapping[str, NormalizedAccountNumber],
    b_map: Mapping[str, NormalizedAccountNumber],
) -> AccountNumberMatch:
    """Compute the best bureau pairing by strict account-number level."""

    best_match = AccountNumberMatch("none", "", "", _EMPTY_NORMALIZED, _EMPTY_NORMALIZED)
    best_rank = _LEVEL_RANK[best_match.level]

    for a_bureau in _BUREAUS:
        a_norm = a_map.get(a_bureau, _EMPTY_NORMALIZED)
        for b_bureau in _BUREAUS:
            b_norm = b_map.get(b_bureau, _EMPTY_NORMALIZED)
            level = match_level(a_norm, b_norm)
            rank = _LEVEL_RANK[level]
            if rank > best_rank:
                best_rank = rank
                best_match = AccountNumberMatch(level, a_bureau, b_bureau, a_norm, b_norm)

    return best_match
