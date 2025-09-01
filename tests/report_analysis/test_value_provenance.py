import logging
from backend.core.logic.report_analysis.report_parsing import (
    _assign_std,
    _empty_bureau_map,
    parse_account_block,
    parse_collection_block,
)


def test_aligned_triple_provenance_and_normalization():
    bm = _empty_bureau_map()
    _assign_std(
        bm,
        "payment_status",
        "Pays As Agreed",
        raw_val="Pays As Agreed",
        provenance="aligned",
    )
    val = bm["payment_status"]
    assert val["raw"] == "Pays As Agreed"
    assert val["normalized"] == "Pays As Agreed"
    assert val["provenance"] == "aligned"


def test_fallback_triple_misaligned_normalization_failure(caplog):
    bm = _empty_bureau_map()
    with caplog.at_level(logging.INFO):
        _assign_std(bm, "credit_limit", "N/A", raw_val="N/A", provenance="fallback")
    val = bm["credit_limit"]
    assert val["provenance"] == "fallback"
    assert val["raw"] == "N/A"
    assert val["normalized"] is None
    assert any(
        "NORM: failed key=credit_limit raw='N/A'" in rec.message for rec in caplog.records
    )


def test_footer_parsing_provenance_and_non_numeric_limit():
    lines = [
        "TransUnion Experian Equifax",
        "Field:",
        "TransUnion Account Type Revolving Payment Frequency Monthly Credit Limit 1000",
        "Experian Account Type Installment Payment Frequency Monthly Credit Limit 2000",
        "Equifax Account Type Mortgage Payment Frequency Monthly Credit Limit none",
        "Two-Year Payment History: OK",
    ]
    maps = parse_account_block(lines)
    eq = maps["equifax"]["credit_limit"]
    assert eq["provenance"] == "footer"
    assert eq["raw"] == "none"
    assert eq["normalized"] is None


def test_collection_block_provenance():
    lines = ["Payment Status Pays As Agreed | Charged Off | Unknown"]
    maps = parse_collection_block(lines, bureau_order=["transunion", "experian", "equifax"])
    ex = maps["experian"]["payment_status"]
    assert ex["provenance"] == "collection"
    assert ex["raw"] == "Charged Off"
