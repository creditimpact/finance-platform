from __future__ import annotations

from typing import List, Dict

from backend.core.logic.report_analysis.column_reader import (
    extract_bureau_table,
    detect_bureau_columns,
)


def _tok(text: str, x0: float, y0: float, x1: float | None = None, y1: float | None = None, line: int = 0) -> dict:
    if x1 is None:
        x1 = x0 + max(10.0, float(len(text) * 5))
    if y1 is None:
        y1 = y0 + 10.0
    return {"text": text, "x0": x0, "y0": y0, "x1": x1, "y1": y1, "line": line, "col": None}


def test_single_label_row_high_balance_mapping():
    # Headers
    tokens: List[dict] = []
    tokens += [
        _tok("Transunion", 80, 50, 120, 62, 0),
        _tok("Experian", 280, 50, 320, 62, 0),
        _tok("Equifax", 480, 50, 520, 62, 0),
    ]
    # Label line: High Balance
    tokens.append(_tok("High Balance:", 10, 100, 60, 112, 1))
    # Values on same row within bands
    tokens.append(_tok("$123,000", 110, 100, 170, 112, 1))  # TU
    tokens.append(_tok("--", 310, 100, 320, 112, 1))        # EX placeholder → empty
    tokens.append(_tok("$456", 510, 100, 540, 112, 1))      # EQ

    block = {"layout_tokens": tokens, "meta": {"debug": {}}}
    res = extract_bureau_table(block)

    assert res["transunion"]["high_balance"] == "$123,000"
    assert res["experian"]["high_balance"] == ""
    assert res["equifax"]["high_balance"] == "$456"

    # Ensure all 22 keys exist
    assert len(res["transunion"]) == 22
    assert len(res["experian"]) == 22
    assert len(res["equifax"]) == 22


def test_wrapped_creditor_remarks_merge():
    # Headers
    tokens: List[dict] = []
    tokens += [
        _tok("Transunion", 80, 50, 120, 62, 0),
        _tok("Experian", 280, 50, 320, 62, 0),
        _tok("Equifax", 480, 50, 520, 62, 0),
    ]
    # Label line
    tokens.append(_tok("Creditor Remarks:", 10, 200, 90, 212, 1))
    # Values for EQ split over two wrapped lines within ΔY ≤ 6
    tokens.append(_tok("Closed or paid account/zero balance", 510, 200, 730, 212, 1))
    tokens.append(_tok("Fannie Mae account", 510, 205, 640, 217, 2))

    block = {"layout_tokens": tokens, "meta": {"debug": {}}}
    res = extract_bureau_table(block)

    assert "Closed or paid account/zero balance Fannie Mae account" == res["equifax"]["creditor_remarks"]
    # Others remain empty but keys exist
    assert res["transunion"]["creditor_remarks"] == ""
    assert res["experian"]["creditor_remarks"] == ""
    assert len(res["transunion"]) == 22 and len(res["experian"]) == 22 and len(res["equifax"]) == 22


def test_detect_bureau_columns_non_overlapping_and_ordered():
    # Synthetic header tokens placed left→right
    toks = [
        _tok("Transunion", 60, 40, 120, 55, 0),
        _tok("Experian", 260, 40, 320, 55, 0),
        _tok("Equifax", 460, 40, 520, 55, 0),
    ]
    cols = detect_bureau_columns(toks)
    assert set(cols.keys()) == {"transunion", "experian", "equifax"}
    # Extract centers and ensure increasing order
    centers = {k: (v[0] + v[1]) / 2.0 for k, v in cols.items()}
    ordered = sorted(centers.items(), key=lambda x: x[1])
    names = [k for k, _ in ordered]
    assert names == ["transunion", "experian", "equifax"]
    # Ensure non-overlap with a strict gap
    intervals = [cols[n] for n in names]
    for a, b in zip(intervals, intervals[1:]):
        assert a[1] <= b[0]  # no overlap (0.5px gap enforced inside implementation)
