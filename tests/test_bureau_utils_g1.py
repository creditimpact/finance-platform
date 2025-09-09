from backend.core.logic.report_analysis._bureau_parse_utils import (
    find_label_groups,
    is_bureau,
    norm,
)


def test_g1_groups_detected():
    lines = [
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
        "TransUnion",
        "1234",
        "Experian",
        "5678",
        "Equifax",
        "9012",
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

    groups = find_label_groups(lines)
    kinds = {g["type"] for g in groups}
    assert "top" in kinds and "bottom" in kinds

    bureaus = {is_bureau(s) for s in lines}
    assert {"transunion", "experian", "equifax"}.issubset(bureaus)
