from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Tradeline:
    """Single bureau tradeline entry."""

    creditor: str
    bureau: str
    account_number: Optional[str] = None
    data: Dict[str, object] = field(default_factory=dict)


@dataclass
class Mismatch:
    """Mismatch for a particular field across bureaus."""

    field: str
    values: Dict[str, object]


@dataclass
class TradelineFamily:
    """Collection of tradelines believed to represent the same account."""

    account_number: str
    tradelines: Dict[str, Tradeline] = field(default_factory=dict)
    mismatches: List[Mismatch] = field(default_factory=list)
