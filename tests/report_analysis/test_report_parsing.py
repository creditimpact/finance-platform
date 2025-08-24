import pytest

from backend.core.logic.report_analysis.report_parsing import (
    extract_payment_statuses,
    extract_account_numbers,
)
from backend.core.logic.utils.names_normalization import normalize_creditor_name


def test_extract_payment_statuses_smartcredit_table():
    text = """
PALISADES FU
Account #            123             123             123
Account Status:      Open            Open            Open
Payment Status:      Collection/Chargeoff  Collection/Chargeoff  Late 120 Days
TransUnion  30:0 60:0 90:0
Experian    30:0 60:0 90:0
Equifax     30:0 60:0 90:0
"""
    statuses = extract_payment_statuses(text)
    key = normalize_creditor_name("PALISADES FU")
    assert statuses[key] == {
        "TransUnion": "Collection/Chargeoff",
        "Experian": "Collection/Chargeoff",
        "Equifax": "Late 120 Days",
    }


@pytest.mark.parametrize(
    "label",
    ["Account #", "Account Number", "Acct #", "Account No."],
)
def test_extract_account_numbers_variants(label):
    text = f"""
PALISADES FU
{label}            1234-56            7890            1357
Balance: 0
"""
    numbers = extract_account_numbers(text)
    key = normalize_creditor_name("PALISADES FU")
    assert numbers[key] == {
        "TransUnion": "123456",
        "Experian": "7890",
        "Equifax": "1357",
    }
