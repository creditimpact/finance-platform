import json
import pytest

from backend.core.case_store import api, storage
from backend.core.logic.report_analysis.extractors import accounts


def setup_case(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "CASESTORE_DIR", tmp_path.as_posix())
    case = api.create_session_case("sess")
    api.save_session_case(case)
    return case.session_id


def test_raw_block_attached_to_case(tmp_path, monkeypatch):
    session_id = setup_case(tmp_path, monkeypatch)
    lines = [
        "JPMCB CARD",
        "Account # 426290**********",
        "Weird Label: FooBar",
        "Payment Status: Current",
    ]
    res = accounts.extract(lines, session_id=session_id, bureau="TransUnion")
    assert res[0]["raw_block"] == "\n".join(lines)
    assert res[0]["fields"]["payment_status"] == "Current"
    assert res[0]["fields"]["extra_fields"]["weird label"] == "FooBar"


def test_unknown_labels_preserved_multiple(tmp_path, monkeypatch):
    session_id = setup_case(tmp_path, monkeypatch)
    lines = [
        "BANK",
        "Account # 123456789",
        "Some New Field: A",
        "Another-Thing: B",
    ]
    res = accounts.extract(lines, session_id=session_id, bureau="Experian")
    extra = res[0]["fields"]["extra_fields"]
    assert extra["some new field"] == "A"
    assert extra["another-thing"] == "B"


def test_no_extra_fields_key_when_all_known(tmp_path, monkeypatch):
    session_id = setup_case(tmp_path, monkeypatch)
    lines = [
        "AMEX",
        "Account # 123456789",
        "Payment Status: Current",
    ]
    res = accounts.extract(lines, session_id=session_id, bureau="Equifax")
    assert res[0]["fields"].get("extra_fields") is None


def test_raw_block_integration_with_persistence(tmp_path, monkeypatch):
    session_id = setup_case(tmp_path, monkeypatch)
    lines = [
        "AMEX",
        "Account # 349992**********",
        "Glitchy Field: XYZ",
        "Payment Status: Current",
    ]
    res = accounts.extract(lines, session_id=session_id, bureau="TransUnion")
    account_id = res[0]["account_id"]
    path = tmp_path / f"{session_id}.json"
    data = json.loads(path.read_text())
    entry = data["accounts"][account_id]
    assert entry["fields"]["raw_block"] == "\n".join(lines)
    assert entry["fields"]["extra_fields"]["glitchy field"] == "XYZ"
