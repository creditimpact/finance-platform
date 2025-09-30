from dataclasses import replace

from backend.core.logic.report_analysis.extractors import accounts
from backend.core.logic.report_analysis.extractors.tokens import parse_date_any


def test_parse_date_any_iso():
    assert parse_date_any("2019-08-01") == "2019-08-01"


def test_parse_date_any_dots():
    assert parse_date_any("20.11.2021") == "2021-11-20"


def test_parse_date_any_slashes_single_digit():
    assert parse_date_any("3/9/2020") == "2020-03-09"


def test_parse_date_any_slashes_zero_padded():
    assert parse_date_any("03/09/2020") == "2020-03-09"


def test_parse_date_any_spaces():
    assert parse_date_any("10 8 2025") == "2025-10-08"


def test_parse_date_any_hyphen_single_digit():
    assert parse_date_any("3-9-2020") == "2020-03-09"


def test_extractor_maps_date_opened_any_format(monkeypatch):
    monkeypatch.setattr(accounts, "upsert_account_fields", lambda **kwargs: None)
    off_flags = replace(
        accounts.FLAGS,
        one_case_per_account_enabled=False,
        normalized_overlay_enabled=False,
    )
    monkeypatch.setattr(accounts, "FLAGS", off_flags)
    lines = [
        "JPMCB CARD",
        "Account # 426290**********",
        "Date Opened: 20.11.2021",
    ]
    res = accounts.extract(lines, session_id="sess", bureau="TransUnion")
    assert res[0]["fields"]["date_opened"] == "2021-11-20"
