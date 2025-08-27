from backend.core.case_store import api, storage
from backend.core.logic.report_analysis.extractors import summary


def setup_case(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "CASESTORE_DIR", tmp_path.as_posix())
    case = api.create_session_case("sess")
    api.save_session_case(case)
    return case.session_id


def test_summary_extraction(tmp_path, monkeypatch):
    session_id = setup_case(tmp_path, monkeypatch)
    lines = [
        "Total Accounts: 3",
        "Open Accounts: 2",
        "Closed Accounts: 1",
        "Derogatory: 0",
    ]
    res = summary.extract(lines, session_id=session_id)
    case = api.load_session_case(session_id)
    assert case.summary.total_accounts == 3
    assert case.summary.open_accounts == 2
    assert case.summary.closed_accounts == 1
    assert case.summary.derogatory == 0
    assert res["total_accounts"] == 3
