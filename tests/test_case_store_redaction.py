import json

from backend.core.case_store.redaction import (
    PII_REPLACEMENTS,
    redact_account_fields,
    redact_for_ai,
)


def test_account_number_last4():
    fields = {"account_number": "1234 5678 9012"}
    redacted = redact_account_fields(fields)
    assert redacted["account_number"] == "****9012"

    fields = {"account_number": "****9012"}
    redacted = redact_account_fields(fields)
    assert redacted["account_number"] == "****9012"

    fields = {"account_number": "123"}
    redacted = redact_account_fields(fields)
    assert redacted["account_number"] == "REDACTED_ACCOUNT"


def test_emails_phones_ssn():
    fields = {
        "contact": "john.doe+test@x.com",
        "phone": "(415) 555-1212",
        "ssn1": "123-45-6789",
        "ssn2": "123456789",
    }
    redacted = redact_account_fields(fields)
    assert redacted["contact"] == PII_REPLACEMENTS["EMAIL"]
    assert redacted["phone"] == PII_REPLACEMENTS["PHONE"]
    assert redacted["ssn1"] == PII_REPLACEMENTS["SSN"]
    assert redacted["ssn2"] == PII_REPLACEMENTS["SSN"]


def test_names_and_addresses():
    fields = {
        "full_name": "Jane Q Public",
        "address_line": "12 Main St, Springfield, IL 62704",
    }
    redacted = redact_account_fields(fields)
    assert redacted["full_name"] == PII_REPLACEMENTS["NAME"]
    assert redacted["address_line"] == PII_REPLACEMENTS["ADDRESS"]


def test_date_normalization():
    fields = {
        "date_opened": "03/10/2019",
        "unparseable": "32/13/2020",
    }
    redacted = redact_account_fields(fields)
    assert redacted["date_opened"] == "2019-03-10"
    assert redacted["unparseable"] == "32/13/2020"


def test_preserve_analytic_fields():
    fields = {
        "balance_owed": 5000,
        "credit_limit": 6000,
        "payment_status": "120D late",
        "two_year_payment_history": "1111",
    }
    redacted = redact_account_fields(fields)
    assert redacted["balance_owed"] == 5000
    assert redacted["credit_limit"] == 6000
    assert redacted["payment_status"] == "120D late"
    assert redacted["two_year_payment_history"] == "1111"


def test_redact_for_ai_shape_contents():
    account = {
        "fields": {
            "account_number": "1234 5678 9012",
            "account_status": "Open",
            "payment_status": "Current",
            "creditor_remarks": "OK",
            "account_description": "Credit card",
            "account_rating": "A",
            "past_due_amount": 0,
            "balance_owed": 500,
            "account_type": "Revolving",
            "creditor_type": "Bank",
            "dispute_status": "None",
            "two_year_payment_history": "1111",
            # intentionally missing credit_limit
            "full_name": "Jane Q Public",
            "email": "john@example.com",
        }
    }
    out = redact_for_ai(account)
    assert set(out.keys()) == {"fields", "field_presence_map", "account_last4"}
    assert out["account_last4"] == "9012"
    fields = out["fields"]
    assert "account_status" in fields
    assert "credit_limit" not in fields
    assert out["field_presence_map"]["credit_limit"] is False
    payload = json.dumps(out)
    assert "@" not in payload
    assert "Jane" not in payload


def test_idempotent():
    fields = {"account_number": "1234 5678 9012", "full_name": "Jane Q Public"}
    once = redact_account_fields(fields)
    twice = redact_account_fields(once)
    assert once == twice


def test_non_string_values():
    fields = {"balance_owed": 100, "meta": {"phone": "415-555-1212", "note": None}}
    redacted = redact_account_fields(fields)
    assert redacted["balance_owed"] == 100
    assert redacted["meta"]["phone"] == PII_REPLACEMENTS["PHONE"]
    assert redacted["meta"]["note"] is None
