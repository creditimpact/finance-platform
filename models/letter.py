from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

from .account import Inquiry


@dataclass
class LetterAccount:
    """Account block in the final dispute letter."""

    name: str
    account_number: str
    status: str
    paragraph: Optional[str] = None
    requested_action: Optional[str] = None
    personal_note: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict) -> 'LetterAccount':
        return cls(
            name=data.get('name', ''),
            account_number=data.get('account_number', ''),
            status=data.get('status', ''),
            paragraph=data.get('paragraph'),
            requested_action=data.get('requested_action'),
            personal_note=data.get('personal_note'),
        )

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class LetterContext:
    """Structured information needed to render a letter."""

    client_name: str = ''
    client_address_lines: List[str] = field(default_factory=list)
    bureau_name: str = ''
    bureau_address: str = ''
    date: str = ''
    opening_paragraph: str = ''
    accounts: List[LetterAccount] = field(default_factory=list)
    inquiries: List[Inquiry] = field(default_factory=list)
    closing_paragraph: str = ''
    is_identity_theft: bool = False

    @classmethod
    def from_dict(cls, data: Dict) -> 'LetterContext':
        return cls(
            client_name=data.get('client_name', ''),
            client_address_lines=list(data.get('client_address_lines', []) or []),
            bureau_name=data.get('bureau_name', ''),
            bureau_address=data.get('bureau_address', ''),
            date=data.get('date', ''),
            opening_paragraph=data.get('opening_paragraph', ''),
            accounts=[LetterAccount.from_dict(a) for a in data.get('accounts', [])],
            inquiries=[Inquiry.from_dict(i) for i in data.get('inquiries', [])],
            closing_paragraph=data.get('closing_paragraph', ''),
            is_identity_theft=bool(data.get('is_identity_theft', False)),
        )

    def to_dict(self) -> Dict:
        d = asdict(self)
        d['accounts'] = [a.to_dict() for a in self.accounts]
        d['inquiries'] = [i.to_dict() for i in self.inquiries]
        return d


@dataclass
class LetterArtifact:
    """Rendered artifacts for a letter."""

    html: str
    pdf_path: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict) -> 'LetterArtifact':
        return cls(html=data.get('html', ''), pdf_path=data.get('pdf_path'))

    def to_dict(self) -> Dict:
        return asdict(self)

