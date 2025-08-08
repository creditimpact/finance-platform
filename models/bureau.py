from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional

from .account import Account


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
