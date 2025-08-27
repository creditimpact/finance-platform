"""Shared regex tokens and helpers for deterministic extractors."""
from __future__ import annotations

import re
from typing import Optional

AMOUNT_RE = re.compile(r"[-+]?\$?\d{1,3}(?:,\d{3})*(?:\.\d+)?")
DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")
ACCOUNT_RE = re.compile(r"ac(?:count|ct)\s*(?:#|number)?[:\s]*([A-Za-z0-9]+)", re.I)

# Canonical field mappings for account level extraction
ACCOUNT_FIELD_MAP = {
    "high balance": "high_balance",
    "last verified": "last_verified",
    "date of last activity": "date_of_last_activity",
    "date reported": "date_reported",
    "date opened": "date_opened",
    "balance owed": "balance_owed",
    "closed date": "closed_date",
    "account rating": "account_rating",
    "account description": "account_description",
    "dispute status": "dispute_status",
    "creditor type": "creditor_type",
    "account status": "account_status",
    "payment status": "payment_status",
    "creditor remarks": "creditor_remarks",
    "payment amount": "payment_amount",
    "last payment": "last_payment",
    "term length": "term_length",
    "past due amount": "past_due_amount",
    "account type": "account_type",
    "payment frequency": "payment_frequency",
    "credit limit": "credit_limit",
    "two-year payment history": "two_year_payment_history",
    "days late": "days_late_7y",
}

SUMMARY_FIELD_MAP = {
    "total accounts": "total_accounts",
    "open accounts": "open_accounts",
    "closed accounts": "closed_accounts",
    "delinquent": "delinquent",
    "derogatory": "derogatory",
    "balances": "balances",
    "payments": "payments",
    "public records": "public_records",
    "inquiries": "inquiries_2y",
}

META_FIELD_MAP = {
    "credit report date": "credit_report_date",
    "name": "name",
    "also known as": "also_known_as",
    "date of birth": "dob",
    "current address": "current_address",
    "previous address": "previous_address",
    "employer": "employer",
}


def parse_amount(text: str) -> Optional[float | int]:
    m = AMOUNT_RE.search(text)
    if not m:
        return None
    val = m.group().replace("$", "").replace(",", "")
    try:
        if "." in val:
            return float(val)
        return int(val)
    except ValueError:
        return None


def parse_date(text: str) -> Optional[str]:
    m = DATE_RE.search(text)
    if m:
        return m.group(0)
    return None
