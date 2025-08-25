import io
import json

from backend.api import app as app_module
from backend.api.app import create_app
from backend.core.orchestrators import extract_problematic_accounts_from_report
from backend.core.logic.report_analysis.report_postprocessing import (
    _assign_issue_types,
)
from tests.test_extract_problematic_accounts import _mock_dependencies


class DummyResult:
    def get(self, timeout=None):
        return {}


def test_start_process_success(monkeypatch, tmp_path):
    class DummyTask:
        def delay(self, *a, **k):
            return DummyResult()

    monkeypatch.setattr(app_module, "extract_problematic_accounts", DummyTask())
    called = {}

    def fake_run(client, proofs, flag):
        called["called"] = True

    monkeypatch.setattr(app_module, "run_credit_repair_process", fake_run)

    test_app = create_app()
    client = test_app.test_client()
    data = {
        "email": "a@example.com",
        "file": (io.BytesIO(b"%PDF-1.4"), "test.pdf"),
    }
    resp = client.post(
        "/api/start-process", data=data, content_type="multipart/form-data"
    )
    assert resp.status_code == 200
    payload = json.loads(resp.data)
    assert payload["status"] == "awaiting_user_explanations"
    assert not called.get("called")


def test_start_process_missing_file():
    test_app = create_app()
    client = test_app.test_client()
    resp = client.post(
        "/api/start-process", data={}, content_type="multipart/form-data"
    )
    assert resp.status_code == 400
    payload = json.loads(resp.data)
    assert "Missing file" in payload["message"]


def test_start_process_emits_enriched_fields(monkeypatch, tmp_path):
    sections = {
        "negative_accounts": [
            {
                "name": "Acc1",
                "issue_types": ["late_payment", "charge_off"],
                "account_number": "123456789",
                "original_creditor": "OC",
                "bureaus": [
                    {"bureau": "Experian", "status": "charge off"},
                    {"bureau": "Equifax", "status": "30 days late"},
                ],
                "balance": 1000,
                "past_due": 500,
                "date_opened": "2020-01-01",
                "date_closed": "2021-01-01",
                "last_activity": "2023-01-01",
            }
        ]
    }
    for acc in sections["negative_accounts"]:
        _assign_issue_types(acc)
    _mock_dependencies(monkeypatch, sections)
    payload = extract_problematic_accounts_from_report("dummy.pdf").to_dict()

    class DummyResult:
        def get(self, timeout=None):
            return payload

    class DummyTask:
        def delay(self, *a, **k):
            return DummyResult()

    monkeypatch.setattr(app_module, "extract_problematic_accounts", DummyTask())
    monkeypatch.setattr(app_module, "run_credit_repair_process", lambda *a, **k: None)
    monkeypatch.setattr(app_module, "set_session", lambda *a, **k: None)

    test_app = create_app()
    client = test_app.test_client()
    data = {
        "email": "a@example.com",
        "file": (io.BytesIO(b"%PDF-1.4"), "test.pdf"),
    }
    resp = client.post(
        "/api/start-process", data=data, content_type="multipart/form-data"
    )
    assert resp.status_code == 200
    payload_json = json.loads(resp.data)
    acc = payload_json["accounts"]["negative_accounts"][0]
    assert acc["primary_issue"] == "charge_off"
    assert acc["account_number_last4"] == "6789"
    assert acc["original_creditor"] == "OC"
    assert acc["bureau_statuses"] == {
        "Experian": "Collection/Chargeoff",
        "Equifax": "30d late",
    }
    assert acc["balance"] == 1000
    assert acc["past_due"] == 500
    assert acc["date_opened"] == "2020-01-01"
    assert acc["date_closed"] == "2021-01-01"
    assert acc["last_activity"] == "2023-01-01"


def test_emitted_account_logs_payment_statuses(monkeypatch, caplog):
    sections = {
        "negative_accounts": [
            {
                "name": "Acc1",
                "payment_statuses": {"TransUnion": "Collection/Chargeoff"},
                "late_payments": {"TransUnion": {"30": 1}},
            }
        ]
    }
    for acc in sections["negative_accounts"]:
        _assign_issue_types(acc)
    _mock_dependencies(monkeypatch, sections)
    with caplog.at_level("INFO", logger="backend.core.orchestrators"):
        extract_problematic_accounts_from_report("dummy.pdf")
    assert any(
        "emitted_account" in r.message and "payment_statuses={'TransUnion': 'Collection/Chargeoff'}" in r.message
        for r in caplog.records
    )


def test_emitted_account_logs_co_marker(monkeypatch, caplog):
    sections = {
        "negative_accounts": [
            {
                "name": "Acc1",
                "late_payments": {"Experian": {"30": 1}},
                "grid_history_raw": {"Experian": "OK CO"},
            }
        ]
    }
    for acc in sections["negative_accounts"]:
        _assign_issue_types(acc)
    _mock_dependencies(monkeypatch, sections)
    with caplog.at_level("INFO", logger="backend.core.orchestrators"):
        payload = extract_problematic_accounts_from_report("dummy.pdf")
    assert any(
        "emitted_account" in r.message
        and "has_co_marker=True" in r.message
        and "co_bureaus=['Experian']" in r.message
        for r in caplog.records
    )
    assert payload.disputes[0].extras.get("primary_issue") == "charge_off"
