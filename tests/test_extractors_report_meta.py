from backend.core.case_store import api, storage
from backend.core.case_store import api
from backend.core.logic.report_analysis.extractors import report_meta


def setup_case(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "CASESTORE_DIR", tmp_path.as_posix())
    case = api.create_session_case("sess")
    api.save_session_case(case)
    return case.session_id


def test_report_meta_extraction(tmp_path, monkeypatch):
    session_id = setup_case(tmp_path, monkeypatch)
    lines = [
        "Name: John Doe",
        "Date of Birth: 1980-01-01",
        "Current Address: 123 Main St",
        "Credit Report Date: 2024-05-01",
    ]
    res = report_meta.extract(lines, session_id=session_id)
    case = api.load_session_case(session_id)
    assert str(case.report_meta.credit_report_date) == "2024-05-01"
    pi = case.report_meta.personal_information
    assert pi.name == "John Doe"
    assert str(pi.dob) == "1980-01-01"
    assert pi.current_address == "123 Main St"
    assert res["credit_report_date"] == "2024-05-01"
