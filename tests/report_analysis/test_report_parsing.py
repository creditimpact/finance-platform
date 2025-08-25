import pytest

from backend.core.logic.report_analysis.report_parsing import (
    extract_account_numbers,
    extract_payment_statuses,
)
from backend.core.logic.utils.names_normalization import normalize_creditor_name


@pytest.mark.parametrize(
    "header_line,payment_line,expected_map",
    [
        (
            "TransUnion          Experian          Equifax",
            "Payment Status:      Collection/Chargeoff  Collection/Chargeoff  Late 120 Days",
            {
                "Transunion": "collection/chargeoff",
                "Experian": "collection/chargeoff",
                "Equifax": "late 120 days",
            },
        ),
        (
            "TransUnion          Experian          Equifax",
            "Payment Status: Collection/Chargeoff   Collection/Chargeoff       Late 120 Days",
            {
                "Transunion": "collection/chargeoff",
                "Experian": "collection/chargeoff",
                "Equifax": "late 120 days",
            },
        ),
        (
            "transunion          EXPERIAN          equifax",
            "Payment Status: COLLECTION/CHARGEOFF  LATE 120 DAYS  Charge-Off",
            {
                "Transunion": "collection/chargeoff",
                "Experian": "late 120 days",
                "Equifax": "charge-off",
            },
        ),
    ],
)
def test_extract_payment_statuses_smartcredit_table(header_line, payment_line, expected_map):
    text = f"""
PALISADES FU
{header_line}
Account #            123             123             123
{payment_line}
"""
    statuses, raw_map = extract_payment_statuses(text)
    key = normalize_creditor_name("PALISADES FU")
    assert statuses[key] == expected_map
    assert (
        raw_map[key]
        == payment_line.split("Payment Status:", 1)[1].strip()
    )


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


def test_extract_account_numbers_masked_forms():
    text = """
PALISADES FU
Account #            ****1234            12 34 56            1234-5678-9012
Balance: 0
"""
    numbers = extract_account_numbers(text)
    key = normalize_creditor_name("PALISADES FU")
    assert numbers[key] == {
        "TransUnion": "****1234",
        "Experian": "123456",
        "Equifax": "123456789012",
    }


def test_extract_account_numbers_skips_non_digits():
    text = """
PALISADES FU
Account #            t disputed            ****1234            N/A
Balance: 0
"""
    numbers = extract_account_numbers(text)
    key = normalize_creditor_name("PALISADES FU")
    assert numbers[key] == {"Experian": "****1234"}
