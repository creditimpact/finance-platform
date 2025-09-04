import pytest

from backend.core.logic.report_analysis.extractors import accounts


def _parse(lines):
    blocks = accounts._split_blocks(lines)
    assert blocks, "No blocks returned"
    return accounts._parse_block(blocks[0])[1]


def test_issuer_captured_from_heading_precedes_account_line():
    lines = [
        "JPMCB CARD",
        "Account # 426290**********",
        "Date Opened: 1.4.1995  1.4.1995  1.4.1995",
    ]
    blocks = accounts._split_blocks(lines)
    assert blocks[0][0] == "__ISSUER_HEADING__: JPMCB CARD"
    fields = _parse(lines)
    assert fields["issuer"] == "JPMCB CARD"


def test_issuer_normalization_uppercase_and_trim():
    lines = [
        " Bk of Amer, ",
        "Account # 123456",
    ]
    fields = _parse(lines)
    assert fields["issuer"] == "BK OF AMER"


def test_no_heading_line_fallback():
    lines = [
        "Account # 123456",
        "Date Opened: 2020-01-01",
    ]
    fields = _parse(lines)
    assert "issuer" not in fields


def test_issuer_not_overwritten_by_creditor_type():
    lines = [
        "AMEX",
        "Account # 123456",
        "Creditor Type: Bank Credit Cards",
    ]
    fields = _parse(lines)
    assert fields["issuer"] == "AMEX"
    assert fields["creditor_type"] == "Bank Credit Cards"
