import pytest

from backend.core.case_store import api, storage
from backend.core.logic.report_analysis.extractors import accounts


def setup_case(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "CASESTORE_DIR", tmp_path.as_posix())
    case = api.create_session_case("sess")
    api.save_session_case(case)
    return case.session_id


def test_account_extraction(tmp_path, monkeypatch):
    session_id = setup_case(tmp_path, monkeypatch)
    lines = [
        "Account # 123456789",
        "Balance Owed: $100",
        "",
        "Acct # 987654321",
        "High Balance: $500",
        "Credit Limit: $1,000",
    ]
    res = accounts.extract(lines, session_id=session_id, bureau="Experian")
    assert len(res) == 2
    case = api.load_session_case(session_id)
    assert case.accounts["6789"].fields.balance_owed == 100
    assert case.accounts["4321"].fields.high_balance == 500
    assert case.accounts["4321"].fields.credit_limit == 1000
    assert case.accounts["6789"].fields.credit_limit is None
