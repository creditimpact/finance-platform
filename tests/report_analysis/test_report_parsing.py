import pytest

from backend.core.logic.report_analysis.report_parsing import (
    extract_account_numbers,
    extract_payment_statuses,
    _DETAIL_LABELS,
    _normalize_detail_value,
)
from backend.core.logic.utils.names_normalization import normalize_creditor_name
from tests.report_analysis.fixtures.account_detail_blocks import (
    BLOCK_WITH_DIGITS,
    BLOCK_WITH_MASKED,
    BLOCK_WITHOUT_DIGITS,
    BLOCK_WITH_COLLECTION_STATUS,
)


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
    numbers = extract_account_numbers(BLOCK_WITH_MASKED)
    key = normalize_creditor_name("PALISADES FU")
    assert numbers[key] == {
        "TransUnion": "****1234",
        "Experian": "123456",
        "Equifax": "123456789012",
    }


def test_extract_account_numbers_skips_non_digits():
    numbers = extract_account_numbers(BLOCK_WITHOUT_DIGITS)
    key = normalize_creditor_name("PALISADES FU")
    assert key not in numbers


def test_account_detail_block_digits_fixture():
    numbers = extract_account_numbers(BLOCK_WITH_DIGITS)
    key = normalize_creditor_name("PALISADES FU")
    assert numbers[key] == {
        "TransUnion": "123",
        "Experian": "456",
        "Equifax": "789",
    }


def test_account_detail_block_collection_status():
    statuses, _ = extract_payment_statuses(BLOCK_WITH_COLLECTION_STATUS)
    key = normalize_creditor_name("PALISADES FU")
    assert statuses[key] == {
        "Transunion": "collection/chargeoff",
        "Experian": "collection/chargeoff",
        "Equifax": "charge-off",
    }


def _label_to_key(label: str) -> str | None:
    for key, regex in _DETAIL_LABELS:
        if regex.search(label.lower()):
            return key
    return None


def test_detail_label_variants_and_account_number_normalization():
    assert _label_to_key("Acct #") == "account_number"
    val, _ = _normalize_detail_value("account_number", "**** 1234")
    assert val == "****1234"
    val2, _ = _normalize_detail_value("account_number", "1234-5678-9012")
    assert val2 == "1234-5678-9012"
    val3, _ = _normalize_detail_value("account_number", "N/A")
    assert val3 is None


def test_normalize_date_formats():
    assert _normalize_detail_value("date_opened", "1/2/23")[0] == "2023-01-02"
    assert _normalize_detail_value("date_opened", "03/2023")[0] == "2023-03"
    assert _normalize_detail_value("date_opened", "Jan 2020")[0] == "2020-01"
