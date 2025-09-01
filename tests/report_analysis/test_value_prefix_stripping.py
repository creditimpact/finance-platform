import pytest

from backend.core.logic.report_analysis.report_parsing import (
    parse_account_block,
    _strip_leaked_prefix,
)


def test_value_prefix_stripping_payment_status():
    lines = [
        "Field: TransUnion Experian Equifax",
        "Payment Status: payment status: collection/chargeoff | payment status: paid",
    ]
    result = parse_account_block(lines)
    assert result["transunion"]["payment_status"] == "collection/chargeoff"
    assert result["experian"]["payment_status"] == "paid"


def test_value_prefix_stripping_account_status():
    lines = [
        "Field: TransUnion Experian Equifax",
        "Account Status: account status: open | account status: closed",
    ]
    result = parse_account_block(lines)
    assert result["transunion"]["account_status"] == "open"
    assert result["experian"]["account_status"] == "closed"


def test_value_prefix_stripping_creditor_remarks():
    lines = [
        "Field: TransUnion Experian Equifax",
        "Creditor Remarks: creditor remarks: charged off | creditor remarks: note",
    ]
    result = parse_account_block(lines)
    assert result["transunion"]["creditor_remarks"] == "charged off"
    assert result["experian"]["creditor_remarks"] == "note"


def test_no_overstrip_unrelated_fields():
    assert (
        _strip_leaked_prefix("balance_owed", "payment status: 100")
        == "payment status: 100"
    )


def test_footer_fields_not_stripped_unless_prefixed():
    lines = [
        "Field: TransUnion Experian Equifax",
        "High Balance: 100 100 100",
        "TransUnion Account Type: Revolving Payment Frequency: Monthly Credit Limit: 1000",
        "Experian Account Type: payment status: charged off Payment Frequency: Monthly Credit Limit: 2000",
        "Equifax Account Type: Installment Payment Frequency: Weekly Credit Limit: 3000",
        "Two-Year Payment History",
    ]
    result = parse_account_block(lines)
    assert result["transunion"]["account_type"] == "revolving"
    assert result["transunion"]["payment_frequency"] == "monthly"
    assert result["experian"]["account_type"] == "charged off"
    assert result["experian"]["credit_limit"] == 2000
    assert result["equifax"]["credit_limit"] == 3000
