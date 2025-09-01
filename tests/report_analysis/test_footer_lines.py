from backend.core.logic.report_analysis.report_parsing import parse_three_footer_lines


def test_footer_handles_missing_middle_and_non_numeric_limit():
    lines = [
        "Account Type: Revolving Payment Frequency: Monthly Credit Limit: $1,500",
        "Account Type: Collection/Chargeoff",
        "Two-Year Payment History: OK",
    ]
    result = parse_three_footer_lines(lines)

    assert result["transunion"]["account_type"] == "revolving"
    assert result["transunion"]["payment_frequency"] == "monthly"
    assert result["transunion"]["credit_limit"] == 1500

    assert result["experian"]["account_type"] is None
    assert result["experian"]["credit_limit"] is None

    assert result["equifax"]["account_type"] == "collection/chargeoff"
    assert result["equifax"]["credit_limit"] is None
