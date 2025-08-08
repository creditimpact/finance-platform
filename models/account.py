from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, TypeVar, Type


AccountId = str
AccountMap = Dict[AccountId, 'Account']


@dataclass
class LateHistory:
    """History of late payments for an account.

    Attributes
    ----------
    date: str
        Date of the late payment in ISO format (YYYY-MM-DD).
    status: str
        Status of the payment at that date (e.g., '30', '60').
    """

    date: str
    status: str

    @classmethod
    def from_dict(cls: Type['LateHistory'], data: Dict) -> 'LateHistory':
        return cls(date=data.get('date', ''), status=data.get('status', ''))

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class Inquiry:
    """Credit inquiry record.

    Attributes
    ----------
    creditor_name: str
        Name of the creditor who made the inquiry.
    date: str
        Date of the inquiry (YYYY-MM-DD).
    bureau: Optional[str]
        Name of the bureau reporting the inquiry.
    """

    creditor_name: str
    date: str
    bureau: Optional[str] = None

    @classmethod
    def from_dict(cls: Type['Inquiry'], data: Dict) -> 'Inquiry':
        return cls(
            creditor_name=data.get('creditor_name', ''),
            date=data.get('date', ''),
            bureau=data.get('bureau'),
        )

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class Account:
    """Representation of a tradeline/account on a credit report.

    Attributes
    ----------
    account_id: Optional[str]
        Identifier used internally for the account.
    name: str
        Creditor or account name.
    account_number: Optional[str]
        Account number as shown on the report.
    reported_status: Optional[str]
        Status reported by the bureau (e.g., 'Open', 'Closed').
    status: Optional[str]
        Additional status text if present.
    dispute_type: Optional[str]
        Type of dispute associated with the account.
    advisor_comment: Optional[str]
        Free form comment from strategist or advisor.
    action_tag: Optional[str]
        Normalized action tag from strategist (e.g., 'dispute').
    recommended_action: Optional[str]
        Human readable action recommendation.
    flags: Optional[List[str]]
        Miscellaneous flags passed through the pipeline.
    """

    name: str
    account_id: Optional[str] = None
    account_number: Optional[str] = None
    reported_status: Optional[str] = None
    status: Optional[str] = None
    dispute_type: Optional[str] = None
    advisor_comment: Optional[str] = None
    action_tag: Optional[str] = None
    recommended_action: Optional[str] = None
    flags: Optional[List[str]] = field(default_factory=list)
    extras: Dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_dict(cls: Type['Account'], data: Dict) -> 'Account':
        return cls(
            account_id=data.get('account_id'),
            name=data.get('name', ''),
            account_number=data.get('account_number'),
            reported_status=data.get('reported_status') or data.get('status'),
            status=data.get('status'),
            dispute_type=data.get('dispute_type'),
            advisor_comment=data.get('advisor_comment'),
            action_tag=data.get('action_tag'),
            recommended_action=data.get('recommended_action'),
            flags=list(data.get('flags', []) or []),
            extras={k: v for k, v in data.items() if k not in {
                'account_id', 'name', 'account_number', 'reported_status',
                'status', 'dispute_type', 'advisor_comment', 'action_tag',
                'recommended_action', 'flags'
            }},
        )

    def to_dict(self) -> Dict:
        data = asdict(self)
        data.update(self.extras)
        return data

