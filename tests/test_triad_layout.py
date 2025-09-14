import logging

from backend.core.logic.report_analysis.triad_layout import (
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
    assert layout.label_band == (0.0, 194.0)
    assert layout.tu_band == (194.0, 275.0)
    assert layout.xp_band == (275.0, 425.0)
    assert layout.eq_band == (425.0, float("inf"))
    assert any(
        rec.message
        == "TRIAD_LAYOUT page=1 label=(0.0,194.0) tu=(194.0,275.0) xp=(275.0,425.0) eq=(425.0,inf)"
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
    assert layout.label_band == (0.0, 194.0)
    assert layout.tu_band == (194.0, 275.0)
    assert layout.xp_band == (275.0, 425.0)
    assert layout.eq_band == (425.0, float("inf"))


def test_assign_band_midpoint_and_tolerance():
    layout = bands_from_header_tokens(
        [
            {"text": "TransUnion", "x0": 0, "x1": 20, "page": 1},
            {"text": "Experian", "x0": 30, "x1": 50, "page": 1},
            {"text": "Equifax", "x0": 60, "x1": 80, "page": 1},
        ]
    )
    # Token slightly to the right of the TU midpoint → ``tu``
    assert assign_band({"x0": 10.1, "x1": 10.2}, layout) == "tu"
    # Clearly within the label band
    assert assign_band({"x0": 1.0, "x1": 1.2}, layout) == "label"
    # Far outside all bands on the left → ``none``
    assert assign_band({"x0": -5.0, "x1": -4.5}, layout) == "none"


def test_assign_band_prefers_xp_on_seam():
    layout = bands_from_header_tokens(
        [
            {"text": "TransUnion", "x0": 160, "x1": 240, "page": 1},
            {"text": "Experian", "x0": 300, "x1": 400, "page": 1},
            {"text": "Equifax", "x0": 460, "x1": 540, "page": 1},
        ]
    )
    # Token on the TU/XP boundary should be assigned to XP
    assert assign_band({"x0": 274.0, "x1": 276.0}, layout) == "xp"
