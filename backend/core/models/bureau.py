from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Dict, Type

from .account import Account, Inquiry


@dataclass
class BureauPayload:
    """Structured data for a single bureau.

    Replaces the previous free-form ``dict`` layout used across the codebase.
    """

    disputes: List[Account] = field(default_factory=list)
    goodwill: List[Account] = field(default_factory=list)
    inquiries: List[Inquiry] = field(default_factory=list)
    high_utilization: List[Account] = field(default_factory=list)

    @classmethod
    def from_dict(cls: Type["BureauPayload"], data: Dict[str, Any]) -> "BureauPayload":
        return cls(
            disputes=[Account.from_dict(d) if isinstance(d, dict) else d for d in data.get("disputes", [])],
            goodwill=[Account.from_dict(d) if isinstance(d, dict) else d for d in data.get("goodwill", [])],
            inquiries=[Inquiry.from_dict(i) if isinstance(i, dict) else i for i in data.get("inquiries", [])],
            high_utilization=[Account.from_dict(d) if isinstance(d, dict) else d for d in data.get("high_utilization", [])],
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "disputes": [a.to_dict() for a in self.disputes],
            "goodwill": [a.to_dict() for a in self.goodwill],
            "inquiries": [i.to_dict() for i in self.inquiries],
            "high_utilization": [a.to_dict() for a in self.high_utilization],
        }


@dataclass
class BureauAccount(Account):
    """Account entry associated with a specific credit bureau."""

    bureau: Optional[str] = None
    section: Optional[str] = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BureauAccount":
        base = Account.from_dict(data)
        return cls(
            **base.to_dict(),
            bureau=data.get("bureau"),
            section=data.get("section"),
        )

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d.update({"bureau": self.bureau, "section": self.section})
        return d


@dataclass
class BureauSection:
    """Collection of accounts belonging to a report section."""

    name: str
    accounts: List[BureauAccount] = field(default_factory=list)

    @classmethod
    def from_dict(cls, name: str, data: List[dict[str, Any]]) -> "BureauSection":
        return cls(name=name, accounts=[BureauAccount.from_dict(d) for d in data])

    def to_dict(self) -> dict[str, Any]:
        return {self.name: [acc.to_dict() for acc in self.accounts]}
