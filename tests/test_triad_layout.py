import logging

from backend.core.logic.report_analysis.triad_layout import (
    TriadLayout,
    assign_band,
    bands_from_header_tokens,
    detect_triads,
)


def test_detect_triads_with_trademark(caplog):
    tokens = [
        {"text": "Transunion\u00ae", "x0": 160, "x1": 240},
        {"text": "Experian\u00ae", "x0": 300, "x1": 400},
        {"text": "Equifax\u00ae", "x0": 460, "x1": 540},
    ]
    tokens_by_line = {(1, 1): tokens}
    with caplog.at_level(logging.INFO):
        layouts = detect_triads(tokens_by_line)
    assert 1 in layouts
    layout = layouts[1]
    assert layout.label_band == (0.0, 200.0)
    assert layout.tu_band == (200.0, 350.0)
    assert layout.xp_band == (350.0, 500.0)
    assert layout.eq_band == (500.0, float("inf"))
    assert any(
        rec.message
        == "TRIAD_LAYOUT page=1 label=(0.0,200.0) tu=(200.0,350.0) xp=(350.0,500.0) eq=(500.0,inf)"
        for rec in caplog.records
    )


def test_bands_from_header_tokens_any_order():
    tokens = [
        {"text": "Equifax", "x0": 460, "x1": 540, "page": 1},
        {"text": "TransUnion", "x0": 160, "x1": 240, "page": 1},
        {"text": "Experian", "x0": 300, "x1": 400, "page": 1},
    ]
    layout = bands_from_header_tokens(tokens)
    assert layout.page == 1
    assert layout.label_band == (0.0, 200.0)
    assert layout.tu_band == (200.0, 350.0)
    assert layout.xp_band == (350.0, 500.0)
    assert layout.eq_band == (500.0, float("inf"))


def test_assign_band_midpoint_and_tolerance():
    layout = TriadLayout(
        page=1,
        label_band=(0.0, 10.0),
        tu_band=(10.0, 20.0),
        xp_band=(20.0, 30.0),
        eq_band=(30.0, 40.0),
    )
    # Slightly beyond the seam → still ``label``
    assert assign_band({"x0": 10.02, "x1": 10.16}, layout) == "label"
    # Farther past tolerance → ``tu``
    assert assign_band({"x0": 16.1, "x1": 16.2}, layout) == "tu"
    # Far beyond last band + tolerance → ``none``
    assert assign_band({"x0": 46.5, "x1": 46.6}, layout) == "none"
