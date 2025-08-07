"""Name normalization helpers for creditors and bureaus."""
from __future__ import annotations

BUREAUS = ["Experian", "Equifax", "TransUnion"]

# Allow a few common variations when looking up bureaus
BUREAU_ALIASES = {
    "transunion": "TransUnion",
    "trans union": "TransUnion",
    "tu": "TransUnion",
    "experian": "Experian",
    "exp": "Experian",
    "ex": "Experian",
    "equifax": "Equifax",
    "eq": "Equifax",
    "efx": "Equifax",
}


def normalize_creditor_name(name: str) -> str:
    """Proxy to :func:`generate_goodwill_letters.normalize_creditor_name`."""
    from ..generate_goodwill_letters import normalize_creditor_name as _norm

    return _norm(name)


def normalize_bureau_name(name: str | None) -> str:
    """Return canonical bureau name for various capitalizations/aliases."""
    if not name:
        return ""
    key = name.strip().lower()
    return BUREAU_ALIASES.get(key, name.title())
