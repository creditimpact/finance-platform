from backend.core.logic.report_analysis.report_parsing import (
    parse_account_block,
    parse_collection_block,
)


def test_footer_handles_missing_middle_and_non_numeric_limit():
    lines = [
        "TransUnion Experian Equifax",
        "Field:",
        "TransUnion Account Type Revolving Payment Frequency Monthly Credit Limit $1,500",
        "Equifax Account Type Collection/Chargeoff Payment Frequency Monthly Credit Limit N/A",
        "Two-Year Payment History: OK",
    ]
    result = parse_account_block(lines)

    assert result["transunion"]["credit_limit"]["normalized"] == 1500.0
    assert result["experian"]["account_type"] is None

    eq = result["equifax"]["credit_limit"]
    assert eq["raw"] == "N/A"
    assert eq["normalized"] is None
    assert eq["provenance"] == "footer"
    assert result["equifax"]["account_type"]["raw"] == "collection/chargeoff"


def test_collection_block_footer_parsing():
    lines = [
        "TransUnion Experian Equifax",
        "Field:",
        "Payment Status Pays As Agreed | Charged Off | Unknown",
        "TransUnion Account Type Revolving Payment Frequency Monthly Credit Limit 1000",
        "Equifax Account Type Collection/Chargeoff Payment Frequency Monthly Credit Limit N/A",
        "Two-Year Payment History: OK",
    ]
    result = parse_collection_block(
        lines, bureau_order=["transunion", "experian", "equifax"]
    )

    assert result["transunion"]["credit_limit"]["normalized"] == 1000.0
    assert result["experian"]["account_type"] is None

    eq = result["equifax"]["credit_limit"]
    assert eq["raw"] == "N/A"
    assert eq["normalized"] is None
    assert eq["provenance"] == "footer"
    assert result["equifax"]["account_type"]["raw"] == "collection/chargeoff"
