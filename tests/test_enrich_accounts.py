import os

from backend.core.logic.report_analysis.block_exporter import enrich_block


def _mk_block(heading: str, lines: list[str]):
    return {
        "heading": heading,
        "lines": lines,
        "meta": {"block_type": "account"},
        "fields": {},
    }


def test_nstar_top_bottom_tu():
    os.environ["BLOCK_DEBUG"] = "0"

    top_labels = [
        "Account #",
        "High Balance:",
        "Last Verified:",
        "Date of Last Activity:",
        "Date Reported:",
        "Date Opened:",
        "Balance Owed:",
        "Closed Date:",
        "Account Rating:",
        "Account Description:",
        "Dispute Status:",
        "Creditor Type:",
    ]
    bottom_labels = [
        "Account Status:",
        "Payment Status:",
        "Creditor Remarks:",
        "Payment Amount:",
        "Last Payment:",
        "Term Length:",
        "Past Due Amount:",
        "Account Type:",
        "Payment Frequency:",
        "Credit Limit:",
    ]

    tu_top = [
        "658671***",
        "$123,000",
        "21.10.2019",
        "21.10.2019",
        "21.10.2019",
        "6.12.2004",
        "$0",
        "21.10.2019",
        "Closed",
        "Joint",
        "Account not disputed",
        "Mortgage Companies - Finance",
    ]
    ex_top = [
        "658671***",
        "$123,000",
        "--",
        "1.10.2019",
        "21.10.2019",
        "1.12.2004",
        "$0",
        "--",
        "Paid",
        "Joint",
        "Account not disputed",
        "Mortgage Companies - Finance",
    ]
    eq_top = [
        "658671***",
        "$123,000",
        "--",
        "1.10.2019",
        "1.10.2019",
        "1.12.2004",
        "$0",
        "--",
        "Paid",
        "Joint",
        "Account not disputed",
        "Mortgage Companies - Finance",
    ]

    tu_bot = [
        "Closed",
        "Current",
        "--",
        "$0",
        "21.10.2019",
        "360 Month(s)",
        "$0",
        "Conventional real estate",
        "mortgage",
        "Monthly (every month)",
    ]
    ex_bot = [
        "Closed",
        "Current",
        "--",
        "$0",
        "21.10.2019",
        "360 Month(s)",
        "$0",
        "Conventional real estate",
        "mortgage",
        "--",
    ]
    # Force a multi-line creditor remark for Equifax to verify safe join + stop
    eq_bot = [
        "Closed",
        "Current",
        "Closed or paid account/zero balance Fannie Mae account",
        "$0",
        "1.10.2019",
        "360 Month(s)",
        "$0",
        "Real estate mortgage",
        "--",
        "$0",
    ]

    lines = (
        ["NSTAR/COOPER"]
        + top_labels
        + ["Transunion"]
        + tu_top
        + ["Experian"]
        + ex_top
        + ["Equifax"]
        + eq_top
        + bottom_labels
        + ["Transunion"]
        + tu_bot
        + ["Experian"]
        + ex_bot
        + ["Equifax"]
        + eq_bot
    )

    out = enrich_block(_mk_block("NSTAR/COOPER", lines))
    f_tu = out["fields"]["transunion"]
    f_eq = out["fields"]["equifax"]

    # TOP expectations
    assert f_tu["account_number_display"] == "658671***"
    # money values are normalized by _g4_apply (no $ or commas)
    assert f_tu["high_balance"] == "123000"
    assert f_tu["date_opened"] == "6.12.2004"
    # BOTTOM expectations
    assert f_tu["account_status"] == "Closed"
    assert f_tu["payment_status"].lower() == "current"
    assert "Closed or paid account/zero balance Fannie Mae account" in f_eq["creditor_remarks"]


def test_seterus_partial_then_bottom():
    os.environ["BLOCK_DEBUG"] = "0"

    # partial TOP (only first 3 labels) for TU, then full BOTTOM groups
    top_labels = [
        "Account #",
        "High Balance:",
        "Last Verified:",
    ]

    lines = (
        ["SETERUS INC"]
        + top_labels
        + ["Transunion", "****2222", "$10,000", "21.10.2019"]
        + [
            "Account Status:",
            "Payment Status:",
            "Creditor Remarks:",
            "Payment Amount:",
            "Last Payment:",
            "Term Length:",
            "Past Due Amount:",
            "Account Type:",
            "Payment Frequency:",
            "Credit Limit:",
        ]
        + ["Transunion"]
        + [
            "Closed",
            "Current",
            "Transferred to another lender",
            "$0",
            "11.2.2019",
            "360 Month(s)",
            "$0",
            "Conventional real estate",
            "mortgage",
            "--",
        ]
        + ["Experian"]
        + [
            "Closed",
            "Current",
            "Transferred to another lender",
            "$0",
            "11.2.2019",
            "360 Month(s)",
            "$0",
            "Conventional real estate",
            "mortgage",
            "--",
        ]
        + ["Equifax"]
        + [
            "Closed",
            "Current",
            "Transferred to another lender",
            "$0",
            "11.2.2019",
            "360 Month(s)",
            "$0",
            "Conventional real estate",
            "mortgage",
            "--",
        ]
    )

    out = enrich_block(_mk_block("SETERUS INC", lines))
    f_tu = out["fields"]["transunion"]
    f_ex = out["fields"]["experian"]
    f_eq = out["fields"]["equifax"]

    assert f_tu["account_number_display"].endswith("2222")
    assert f_tu["payment_status"].lower() == "current"
    assert f_tu["account_type"].startswith("Conventional real estate")
    # Ensure padding/cleaning for missing or '--' values in EX/EQ
    assert f_ex["credit_limit"] == ""
    assert f_eq["credit_limit"] == ""
