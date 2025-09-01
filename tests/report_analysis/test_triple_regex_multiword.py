import pytest
from backend.core.logic.report_analysis.report_parsing import TRIPLE_LINE_RE, _split_triple_fallback


def test_triple_aligned_multi_word_values():
    line1 = "Account Status: Paid In Full  Open Current  --"
    m1 = TRIPLE_LINE_RE.match(line1)
    assert m1
    assert m1.group("v1") == "Paid In Full"
    assert m1.group("v2") == "Open Current"
    assert m1.group("v3") == "--"

    line2 = "Creditor Remarks: SOME COMPANY LLC  OTHER BANK NA  --"
    m2 = TRIPLE_LINE_RE.match(line2)
    assert m2
    assert m2.group("v1") == "SOME COMPANY LLC"
    assert m2.group("v2") == "OTHER BANK NA"
    assert m2.group("v3") == "--"


def test_triple_fallback_without_alignment():
    bureau_order = ["transunion", "experian", "equifax"]
    v1, v2, v3 = _split_triple_fallback("Paid In Full -- Open Current", bureau_order)
    assert v1 == "Paid In Full"
    assert v2 is None
    assert v3 == "Open Current"
