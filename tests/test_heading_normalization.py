import logging

from backend.core.logic.report_analysis.analyze_report import _log_heading_join_misses
from backend.core.logic.utils.norm import normalize_heading


def test_normalize_heading_alias():
    assert normalize_heading("AMEX") == "american express"
    assert normalize_heading("GS Bank USA") == "gs"


def test_heading_join_miss_logged(caplog):
    existing = {normalize_heading("GS")}
    heading_map = {normalize_heading("AMEX"): "AMEX"}
    paymap = {normalize_heading("AMEX"): {"Experian": "OK"}}
    with caplog.at_level(logging.INFO):
        _log_heading_join_misses(paymap, "payment_statuses", existing, heading_map)
    assert any("heading_join_miss" in r.message for r in caplog.records)
