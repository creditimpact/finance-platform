import os

from backend.core.logic.report_analysis.block_exporter import enrich_block_v2


def _mk_block(heading: str, lines: list[str]):
    return {
        "heading": heading,
        "lines": lines,
        "meta": {"block_type": "account"},
        "fields": {},
    }


def test_top_bottom_full_account():
    os.environ["BLOCK_DEBUG"] = "0"
    # Minimal stacked layout: TOP labels, then TU/EX/EQ tokens with TOP values,
    # then BOTTOM labels followed by 3x10 values (one chunk per bureau)
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
        "****1234",
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
        "****5678",
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
        "****9012",
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
    # bottom values per bureau in sequence (10 each)
    tu_bot = [
        "Closed",
        "Current",
        "Closed or paid account/zero balance Fannie Mae account",
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
    eq_bot = [
        "Closed",
        "Current",
        "Account transferred or sold",
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
        + tu_bot
        + ex_bot
        + eq_bot
    )
    out = enrich_block_v2(_mk_block("NSTAR/COOPER", lines))
    f = out["fields"]
    # TOP
    assert f["transunion"]["account_number_display"].endswith("1234")
    assert f["experian"]["date_reported"] == "21.10.2019"
    assert f["equifax"]["balance_owed"] == "0"
    # BOTTOM
    assert f["transunion"]["payment_status"].lower() == "current"
    assert "Fannie Mae" in f["transunion"]["creditor_remarks"]
    assert f["equifax"]["account_type"].startswith("Real estate")
    # Presence + tail + money cleaning
    m = out["meta"]
    assert m["bureau_presence"]["transunion"] is True
    assert m["bureau_presence"]["experian"] is True
    assert m["bureau_presence"]["equifax"] is True
    assert m.get("account_number_tail") in {"1234", "5678", "9012"}
    assert f["transunion"]["high_balance"] == "123000"


def test_partial_top_then_bottom_merge():
    os.environ["BLOCK_DEBUG"] = "0"
    # Only TOP for TU, later BOTTOM provides extra data
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
        * 3
    )
    out = enrich_block_v2(_mk_block("SETERUS INC", lines))
    f = out["fields"]["transunion"]
    assert f["account_number_display"].endswith("2222")
    assert f["payment_status"].lower() == "current"
    assert f["account_type"].startswith("Conventional real estate")


def test_presence_tail():
    os.environ["BLOCK_DEBUG"] = "0"
    lines = [
        "AMEX",
        "Account #",
        "High Balance:",
        "Transunion",
        "****9999",
        "$1,000",
        "Account Status:",
        "Payment Status:",
        "Closed",
        "Current",
    ]
    out = enrich_block_v2(_mk_block("AMEX", lines))
    m = out["meta"]
    assert m["account_number_tail"] == "9999"
    assert any(out["fields"]["transunion"].values())


def test_days_late_and_history():
    os.environ["BLOCK_DEBUG"] = "0"
    lines = [
        "BANKAMERICA",
        "Account #",
        "High Balance:",
        "TransUnion",
        "****0000",
        "$5,000",
        "Two-Year Payment History",
        "TransUnion",
        "OK",
        "Jan",
        "OK",
        "Feb",
        "Experian",
        "OK",
        "Equifax",
        "OK",
        "Days Late - 7 Year History",
        "TransUnion",
        "30: 0",
        "60: 0",
        "90: 0",
        "Experian",
        "30: 0",
        "60: 0",
        "90: 0",
        "Equifax",
        "30: 0",
        "60: 0",
        "90: 0",
    ]
    out = enrich_block_v2(_mk_block("BANKAMERICA", lines))
    m = out["meta"]
    ph = m.get("payment_history", {})
    assert "transunion" in ph and len(ph["transunion"]) >= 2
    dl = m.get("days_late_7y", {})
    assert dl.get("transunion") == {"30": "0", "60": "0", "90": "0"}
