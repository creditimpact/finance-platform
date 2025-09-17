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
    assert layout.label_band == (0.0, 160.0)
    assert layout.tu_band == (160.0, 300.0)
    assert layout.xp_band == (300.0, 460.0)
    assert layout.eq_band == (460.0, float("inf"))
    assert any(
        rec.message
        == "TRIAD_LAYOUT page=1 label=(0.0,160.0) tu=(160.0,300.0) xp=(300.0,460.0) eq=(460.0,inf)"
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
    assert layout.label_band == (0.0, 160.0)
    assert layout.tu_band == (160.0, 300.0)
    assert layout.xp_band == (300.0, 460.0)
    assert layout.eq_band == (460.0, float("inf"))


def test_assign_band_midpoint_and_tolerance():
    layout = bands_from_header_tokens(
        [
            {"text": "TransUnion", "x0": 100, "x1": 140, "page": 1},
            {"text": "Experian", "x0": 200, "x1": 240, "page": 1},
            {"text": "Equifax", "x0": 300, "x1": 340, "page": 1},
        ]
    )
    # Token with midpoint within the TU column → ``tu``
    assert assign_band({"x0": 150.0, "x1": 170.0}, layout) == "tu"
    # Clearly within the label band
    assert assign_band({"x0": 10.0, "x1": 20.0}, layout) == "label"
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
    assert assign_band({"x0": 299.0, "x1": 301.0}, layout) == "xp"
