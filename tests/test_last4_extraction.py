import logging

from backend.core.logic.report_analysis.extractors.accounts import extract_last4


def test_extract_last4_happy_path_masked():
    assert extract_last4("Account # 942029*******") == "2029"


def test_extract_last4_with_hyphens_and_mask():
    assert extract_last4("Account # -34999***********") == "4999"


def test_extract_last4_all_masked():
    assert extract_last4("Account # ****") == ""


def test_extract_last4_mixed_spaces():
    assert extract_last4("Account # 4262 90******") == "6290"


def test_extractor_uses_helper(caplog):
    lines = [
        "JPMCB CARD",
        "Account # 942029*******",
        "Date Opened: 2020-01-01",
    ]
    caplog.set_level(logging.DEBUG)
    # Disable case store interactions
    from backend.core.logic.report_analysis.extractors import accounts

    accounts.upsert_account_fields = lambda **kwargs: None
    accounts.extract(lines, session_id="sess_demo", bureau="TransUnion")
    assert any(
        "CASEBUILDER: last4_extracted" in rec.message and "2029" in rec.message
        for rec in caplog.records
    )
