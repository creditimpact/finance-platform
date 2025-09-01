import logging

from backend.core.logic.report_analysis.report_parsing import (
    _assign_std,
    _empty_bureau_map,
    parse_account_block,
    parse_collection_block,
)
from backend.core.materialize.account_materializer import _get_scalar


def test_aligned_triple_preserves_raw_and_provenance():
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


def test_fallback_triple_provenance():
    lines = ["Payment Status Pays As Agreed | Charged Off | Unknown"]
    maps = parse_collection_block(lines, bureau_order=["transunion", "experian", "equifax"])
    ex = maps["experian"]["payment_status"]
    assert ex["provenance"] == "fallback"
    assert ex["raw"] == "Charged Off"


def test_footer_parsing_provenance():
    lines = [
        "TransUnion Experian Equifax",
        "Field:",
        "TransUnion Account Type Revolving Payment Frequency Monthly Credit Limit 1000",
        "Experian Account Type Installment Payment Frequency Monthly Credit Limit 2000",
        "Equifax Account Type Mortgage Payment Frequency Monthly Credit Limit 3000",
    ]
    maps = parse_account_block(lines)
    ex = maps["experian"]["account_type"]
    assert ex["provenance"] == "footer"
    eq = maps["equifax"]["credit_limit"]
    assert eq["normalized"] == 3000.0


def test_normalization_failure_logs_and_keeps_raw(caplog):
    lines = [
        "TransUnion Experian Equifax",
        "Field:",
        "Credit Limit: N/A  2000  3000",
    ]
    with caplog.at_level(logging.INFO):
        maps = parse_account_block(lines)
    tu = maps["transunion"]["credit_limit"]
    assert tu["raw"] == "N/A"
    assert tu["normalized"] is None
    assert any("norm_failed key=credit_limit raw='N/A'" in rec.message for rec in caplog.records)


def test_get_scalar_helper():
    assert _get_scalar({"raw": "10", "normalized": 10}) == 10
    assert _get_scalar({"raw": "10", "normalized": None}) == "10"
    assert _get_scalar("x") == "x"
