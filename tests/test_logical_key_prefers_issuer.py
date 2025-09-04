import logging
import pytest

from backend.core.logic.report_analysis.extractors import accounts
from backend.core.logic.report_analysis.keys import compute_logical_account_key


@pytest.fixture(autouse=True)
def patch_case_store(monkeypatch):
    monkeypatch.setattr(accounts, "upsert_account_fields", lambda **kwargs: None)
    monkeypatch.setattr(accounts, "emit_account_field_coverage", lambda **kwargs: None)
    monkeypatch.setattr(accounts, "emit_session_field_coverage_summary", lambda **kwargs: None)


def _build_lines(include_heading: bool) -> list[str]:
    lines = []
    if include_heading:
        lines.append("JPMCB CARD")
    lines.extend(
        [
            "Account # 00000290",
            "Creditor Type: Bank Credit Cards",
            "Date Opened: 2020-01-01",
        ]
    )
    return lines


def test_logical_key_uses_issuer_when_present(monkeypatch, caplog):
    lines = _build_lines(include_heading=True)
    captured = {}

    def fake_compute(issuer, last4, account_type, date_opened):
        captured["args"] = (issuer, last4, account_type, date_opened)
        return "LK123"

    monkeypatch.setattr(accounts, "compute_logical_account_key", fake_compute)
    caplog.set_level(logging.DEBUG)
    accounts.extract(lines, session_id="sess", bureau="TransUnion")
    assert captured["args"][0] == "JPMCB CARD"


def test_logical_key_falls_back_to_creditor_type_when_no_issuer(monkeypatch, caplog):
    lines = _build_lines(include_heading=False)
    captured = {}

    def fake_compute(issuer, last4, account_type, date_opened):
        captured["args"] = (issuer, last4, account_type, date_opened)
        return "LK456"

    monkeypatch.setattr(accounts, "compute_logical_account_key", fake_compute)
    caplog.set_level(logging.DEBUG)
    accounts.extract(lines, session_id="sess2", bureau="TransUnion")

    assert captured["args"][0] == "Bank Credit Cards"
    messages = [r.message for r in caplog.records if "CASEBUILDER: logical_key" in r.message]
    assert messages, "debug log missing"
    msg = messages[0]
    assert "issuer=None" in msg or "issuer=''" in msg
    assert "creditor_type='Bank Credit Cards'" in msg
    assert "lk='LK456'" in msg


def test_no_regression_when_both_present(monkeypatch, caplog):
    lines = _build_lines(include_heading=True)
    original_compute = accounts.compute_logical_account_key
    captured = {}

    def wrapper(issuer, last4, account_type, date_opened):
        captured["args"] = (issuer, last4, account_type, date_opened)
        return original_compute(issuer, last4, account_type, date_opened)

    monkeypatch.setattr(accounts, "compute_logical_account_key", wrapper)
    expected_lk = original_compute("JPMCB CARD", "0290", None, "2020-01-01")
    caplog.set_level(logging.DEBUG)
    accounts.extract(lines, session_id="sess3", bureau="TransUnion")

    assert captured["args"][0] == "JPMCB CARD"
    messages = [r.message for r in caplog.records if "CASEBUILDER: logical_key" in r.message]
    assert messages
    msg = messages[0]
    assert "issuer='JPMCB CARD'" in msg
    assert f"lk='{expected_lk}'" in msg
