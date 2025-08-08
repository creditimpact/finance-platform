import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pytest
from models.account import Account, Inquiry, LateHistory
from models.bureau import BureauSection, BureauAccount
from models.strategy import StrategyPlan, StrategyItem, Recommendation
from models.letter import LetterContext, LetterAccount, LetterArtifact


def test_account_roundtrip():
    data = {
        'account_id': '1',
        'name': 'ABC Bank',
        'account_number': '1234',
        'reported_status': 'Open',
        'flags': ['x'],
        'extra': 'y',
    }
    acc = Account.from_dict(data)
    assert acc.name == 'ABC Bank'
    back = acc.to_dict()
    assert back['account_id'] == '1'
    assert back['extra'] == 'y'


def test_inquiry_roundtrip():
    data = {'creditor_name': 'XYZ', 'date': '2020-01-01', 'bureau': 'Experian'}
    obj = Inquiry.from_dict(data)
    assert obj.to_dict() == data


def test_letter_context_roundtrip():
    ctx = LetterContext(
        client_name='John',
        client_address_lines=['1 St'],
        bureau_name='Experian',
        bureau_address='Addr',
        date='Today',
        opening_paragraph='Op',
        accounts=[LetterAccount(name='A', account_number='1', status='Open')],
        inquiries=[Inquiry(creditor_name='B', date='2020-01-01')],
        closing_paragraph='Bye',
        is_identity_theft=False,
    )
    assert LetterContext.from_dict(ctx.to_dict()) == ctx
