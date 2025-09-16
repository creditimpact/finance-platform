from backend.core.logic.report_analysis.problem_extractor import (
    build_rule_fields_from_triad,
)


def test_triad_adapter_basic_resolution_and_parsing():
    account = {
        "triad": {"enabled": True, "order": ["experian", "equifax", "transunion"]},
        "triad_fields": {
            "transunion": {
                "payment_status": "Current",
                "account_status": "Closed",
                "past_due_amount": "$0",
                "balance_owed": "$1,234",
                "credit_limit": "$2,500",
                "creditor_remarks": "Closed TU",
                "account_type": "Type TU",
            },
            "experian": {
                "payment_status": "Late",
                "account_status": "Open",
                "past_due_amount": "$12,091",
                "balance_owed": "$1,235",
                "credit_limit": "$2,600",
                "creditor_remarks": "Closed XP",
                "account_type": "Type XP",
            },
            "equifax": {
                "payment_status": "OK",
                "account_status": "Closed",
                "past_due_amount": "$0",
                "balance_owed": "$1,236",
                "credit_limit": "$2,700",
                "creditor_remarks": "Closed EQ",
                "account_type": "Type EQ",
            },
        },
        "seven_year_history": {
            "transunion": {"late30": 1, "late60": 0, "late90": 0},  # sum=1
            "experian": {"late30": 0, "late60": 2, "late90": 0},    # sum=2
            "equifax": {"late30": 0, "late60": 0, "late90": 3},     # sum=3 -> max
        },
        "two_year_payment_history": {
            "transunion": ["OK", "ok", "OK"],
            "experian": ["OK", "30", "OK"],  # non-OK token -> derog
            "equifax": [],
        },
    }

    fields, prov = build_rule_fields_from_triad(account)

    # Numeric parsing
    assert fields["past_due_amount"] == 12091.0
    assert fields["balance_owed"] == 1235.0  # taken from Experian per order
    assert fields["credit_limit"] == 2600.0   # taken from Experian per order

    # Status resolution via triad order
    assert fields["payment_status"] == "Late"
    assert fields["account_status"] == "Open"
    # provenance reflects triad order pick (experian)
    assert prov.get("payment_status") == "experian"
    assert prov.get("account_status") == "experian"

    # History aggregations
    assert fields["days_late_7y"] == 3  # max across bureaus
    assert fields["has_derog_2y"] is True
