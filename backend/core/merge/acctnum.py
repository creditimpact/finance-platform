"""Account-number normalization and bureau-pair matching helpers."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Mapping

__all__ = [
    "AccountNumberMatch",
    "NormalizedAccountNumber",
    "acctnum_level",
    "acctnum_match_visible",
    "best_account_number_match",
    "match_level",
    "normalize_display",
]

_BUREAUS = ("transunion", "experian", "equifax")

_LEVEL_POINTS: Dict[str, int] = {
    "exact_or_known_match": 28,
}

_LEVEL_RANK: Dict[str, int] = {
    "none": 0,
    "exact_or_known_match": 1,
}


@dataclass(frozen=True)
class NormalizedAccountNumber:
    """Normalized representation of a bureau account number display."""

    raw: str
    digits: str

    @property
    def has_digits(self) -> bool:
        return bool(self.digits)

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


def _digits_only(s: str) -> str:
    return "".join(ch for ch in (s or "") if ch.isdigit())


def acctnum_match_visible(a_raw: str, b_raw: str) -> tuple[bool, dict[str, str]]:
    """Implement the visible-digits substring rule."""

    a = _digits_only(a_raw)
    b = _digits_only(b_raw)

    short, long_ = (a, b) if len(a) <= len(b) else (b, a)

    if not short or not long_:
        return False, {"short": short, "long": long_, "why": "empty"}

    if short in long_:
        return True, {"short": short, "long": long_, "why": "substring"}

    return False, {"short": short, "long": long_, "why": "mismatch"}


def acctnum_level(a_raw: str, b_raw: str) -> tuple[str, dict[str, str]]:
    """Return the account-number level and debug metadata."""

    ok, debug = acctnum_match_visible(a_raw, b_raw)
    level = "exact_or_known_match" if ok else "none"
    debug_dict = dict(debug)
    debug_dict.setdefault("short", "")
    debug_dict.setdefault("long", "")
    return level, debug_dict


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
        "none",
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
