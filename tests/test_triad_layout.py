import logging

from backend.core.logic.report_analysis.triad_layout import (
    TriadLayout,
    assign_band,
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
    assert layout.label_band == (0.0, 125.0)
    assert layout.tu_band == (125.0, 275.0)
    assert layout.xp_band == (275.0, 425.0)
    assert layout.eq_band == (425.0, 575.0)
    assert any(
        rec.message
        == "TRIAD_LAYOUT page=1 label=(0.0,125.0) tu=(125.0,275.0) xp=(275.0,425.0) eq=(425.0,575.0)"
        for rec in caplog.records
    )


def test_assign_band_midpoint_and_tolerance():
    layout = TriadLayout(
        page=1,
        label_band=(0.0, 10.0),
        tu_band=(10.0, 20.0),
        xp_band=(20.0, 30.0),
        eq_band=(30.0, 40.0),
    )
    # midpoint exactly on the boundary -> label
    assert assign_band({"x0": 9.5, "x1": 10.5}, layout) == "label"
    # midpoint slightly past boundary but within tolerance -> label
    assert assign_band({"x0": 10.02, "x1": 10.16}, layout) == "label"
    # midpoint beyond tolerance -> tu
    assert assign_band({"x0": 10.2, "x1": 10.3}, layout) == "tu"
    # midpoint beyond last band plus tolerance -> none
    assert assign_band({"x0": 40.2, "x1": 40.3}, layout) == "none"

