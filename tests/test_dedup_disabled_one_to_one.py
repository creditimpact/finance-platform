from backend.core.case_store import api, storage
from backend.core.logic.report_analysis.extractors import accounts
from backend.core.config.flags import Flags


def set_flags(monkeypatch, *, debug=True):
    monkeypatch.setattr(accounts, "FLAGS", Flags(one_case_per_account_enabled=False, casebuilder_debug=debug))
    monkeypatch.setattr(api, "FLAGS", Flags(one_case_per_account_enabled=False))


def setup_case(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "CASESTORE_DIR", tmp_path.as_posix())
    session_id = "sess"
    case = api.create_session_case(session_id)
    api.save_session_case(case)
    return session_id


def test_one_block_one_case_when_dedup_off(tmp_path, monkeypatch):
    set_flags(monkeypatch)
    session_id = setup_case(tmp_path, monkeypatch)
    lines = [
        "Account # 123456789",
        "Balance Owed: $100",
    ]
    blocks = accounts._split_blocks(lines)
    accounts.extract(lines, session_id=session_id, bureau="Experian")
    case = api.load_session_case(session_id)
    assert len(blocks) == len(case.accounts) == 1


def test_multiple_blocks_multiple_cases_when_dedup_off(tmp_path, monkeypatch):
    set_flags(monkeypatch)
    session_id = setup_case(tmp_path, monkeypatch)
    lines = [
        "Account # 123456789",
        "Balance Owed: $100",
        "",
        "Account # 123456789",
        "Balance Owed: $200",
    ]
    blocks = accounts._split_blocks(lines)
    accounts.extract(lines, session_id=session_id, bureau="Experian")
    case = api.load_session_case(session_id)
    assert len(blocks) == 2
    assert len(case.accounts) == 2
