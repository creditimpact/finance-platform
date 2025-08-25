import logging

from backend.core.logic.report_analysis.analyze_report import _join_heading_map
from backend.core.logic.utils.norm import normalize_heading


def test_normalize_heading_alias():
    assert normalize_heading("AMEX") == "american express"
    assert normalize_heading("GS Bank USA") == "gs"
    assert normalize_heading("WEBBNK/FHUT") == "webbank fingerhut"
    assert normalize_heading("CREDITONEBNK") == "credit one bank"


def test_fuzzy_heading_join(caplog):
    accounts = [{"name": "Capital One"}]
    norm = normalize_heading("Capital One")
    accounts_by_norm = {norm: [accounts[0]]}
    existing = {norm}
    paymap = {normalize_heading("CAPTL ONE"): {"Experian": "OK"}}
    heading_map = {normalize_heading("CAPTL ONE"): "CAPTL ONE"}
    with caplog.at_level(logging.INFO):
        _join_heading_map(
            accounts_by_norm,
            existing,
            paymap,
            "payment_statuses",
            heading_map,
            is_bureau_map=True,
            aggregate_field="payment_status",
        )
    assert accounts[0]["payment_statuses"]["Experian"] == "OK"
    assert any(
        "joined_heading" in r.message and '"method": "fuzzy"' in r.message
        for r in caplog.records
    )
