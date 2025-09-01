import pytest
from backend.core.logic.report_analysis.report_parsing import (
    parse_account_block,
    stitch_account_blocks,
)


def test_joiner_merges_wrapped_value():
    lines = [
        "PALISADES FU",
        "Field: TransUnion Experian Equifax",
        "Creditor Remarks: Amount in H/C column is credit",
        "limit",
    ]
    result = parse_account_block(lines)
    assert (
        result["transunion"].get("creditor_remarks")
        == "Amount in H/C column is credit limit"
    )


def test_resume_after_history_preserves_order_and_fields():
    block = """AMERICAN EXPRESS
Field: Experian Equifax TransUnion
Account #:          456             789             123
High Balance:       1000            1000            1000
Last Verified:      2023-01-01      2023-01-01      2023-01-01
Date of Last Activity: 2022-12-01 2022-12-01 2022-12-01
Date Reported:      2023-01-02      2023-01-02      2023-01-02
Date Opened:        2020-01-01      2020-01-01      2020-01-01
Balance Owed:       500             500             500
Account Status:     Open            Open            Open
Payment Amount:     100             100             100
Two-Year Payment History: OK1 OK2 OK3
Payment Status:     EXPS           EQPS           TUPS
Creditor Remarks:   Wrapped line continues
here
Credit Limit:       2000            2000            2000
Past Due Amount:    0               0               0
Account Type:       Revolving       Revolving       Revolving
Payment Frequency:  Monthly         Monthly         Monthly
Last Payment:       2023-03-01      2023-03-01      2023-03-01
"""
    lines = block.splitlines()
    maps_list = stitch_account_blocks(lines)
    assert maps_list, "Expected one account block"
    tu_map = maps_list[0]["transunion"]
    filled = sum(1 for v in tu_map.values() if v is not None)
    assert filled >= 10
    assert tu_map["payment_status"] == "TUPS"
    assert (
        maps_list[0]["experian"]["creditor_remarks"]
        == "Wrapped line continues here"
    )
