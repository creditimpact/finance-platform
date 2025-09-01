from backend.core.logic.report_analysis.report_parsing import stitch_account_blocks


def _parse(lines):
    blocks = stitch_account_blocks(lines)
    assert blocks, "no blocks parsed"
    return blocks[0]


def test_resume_after_history_same_account():
    lines = [
        "ACME BANK",
        "Field: TransUnion Experian Equifax",
        "Balance: 10  20  30",
        "Credit Limit: 40  50  60",
        "Revolving Monthly 1000",
        "Installment Monthly 2000",
        "Mortgage Monthly 3000",
        "Two-Year Payment History",
        "TU history",
        "Days Late -7 Year History",
        "TU 30:0 60:0 90:0",
        "High Balance: 70  80  90",
    ]
    res = _parse(lines)
    assert res["transunion"]["high_balance"] == 70
    assert res["experian"]["high_balance"] == 80


def test_carry_forward_bureau_order_without_header():
    lines = [
        "ACME BANK",
        "Field: TransUnion Experian Equifax",
        "Balance: 10  20  30",
        "Credit Limit: 40  50  60",
        "Revolving Monthly 1000",
        "Installment Monthly 2000",
        "Mortgage Monthly 3000",
        "Two-Year Payment History",
        "TU history",
        "Days Late -7 Year History",
        "TU 30:0 60:0 90:0",
        "High Balance: 70  80  90",
    ]
    res = _parse(lines)
    assert res["experian"]["high_balance"] == 80


def test_footer_and_url_are_ignored():
    lines = [
        "ACME BANK",
        "Field: TransUnion Experian Equifax",
        "Balance: 10  20  30",
        "3-Bureau Credit Report & Scores | SmartCredit",
        "https://www.smartcredit.com/",
        "Revolving Monthly 1000",
        "Installment Monthly 2000",
        "Mortgage Monthly 3000",
        "Two-Year Payment History",
        "TU history",
        "Days Late -7 Year History",
        "TU 30:0 60:0 90:0",
        "High Balance: 70  80  90",
    ]
    res = _parse(lines)
    assert res["transunion"]["high_balance"] == 70
