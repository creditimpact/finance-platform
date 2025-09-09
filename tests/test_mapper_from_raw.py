from __future__ import annotations

from backend.core.logic.report_analysis.mapper_from_raw import map_raw_to_fields


def _row(y, tu, ex, eq):
    return {
        "row_y": y,
        "transunion": [{"text": s} for s in tu.split()] if tu else [],
        "experian": [{"text": s} for s in ex.split()] if ex else [],
        "equifax": [{"text": s} for s in eq.split()] if eq else [],
        "transunion_text": tu,
        "experian_text": ex,
        "equifax_text": eq,
    }


def test_map_raw_to_fields_minimal_rows_and_cleaning():
    # Minimal RAW with 3 rows mapping to first 3 labels
    raw = {
        "rows": [
            _row(100, "****1234", "--", "--"),          # Account #
            _row(110, "$123,000", "$999", "--"),        # High Balance
            _row(120, "21.10.2019", "1.10.2019", "--"),  # Last Verified
        ]
    }
    out = map_raw_to_fields(raw)
    assert set(out.keys()) == {"transunion", "experian", "equifax"}
    # 22 keys present
    assert len(out["transunion"]) == 22
    assert len(out["experian"]) == 22
    assert len(out["equifax"]) == 22
    # Cleaning: '--' becomes empty
    assert out["experian"]["account_number_display"] == ""
    # Values preserved otherwise
    assert out["transunion"]["account_number_display"].endswith("1234")


def test_wrapped_merge_by_row_text():
    # Two fragments on the same row should merge via *_text already in RAW
    raw = {
        "rows": [
            _row(200, "Closed or paid account/zero balance Fannie Mae account", "", ""),
        ]
    }
    out = map_raw_to_fields(raw)
    # Label 0 is Account #; since only one row provided, it maps to first key
    # We mainly assert cleaning and presence of keys
    assert len(out["transunion"]) == 22
    # Placeholder handling checked above; here we ensure the string propagates
    assert "Fannie Mae account" in out["transunion"]["account_number_display"]

