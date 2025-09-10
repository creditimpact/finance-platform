import io
import json

import backend.config as config
import backend.core.orchestrators as orch
from backend.api import app as app_module
from backend.api.app import create_app
from backend.core.logic.report_analysis.report_postprocessing import _assign_issue_types
from backend.core.orchestrators import extract_problematic_accounts_from_report
from tests.test_extract_problematic_accounts import _mock_dependencies


class DummyResult:
    def get(self, timeout=None):
        return {}


def test_start_process_success(monkeypatch, tmp_path):
    monkeypatch.setattr(app_module, "run_full_pipeline", lambda sid: DummyResult())
    monkeypatch.setattr(app_module.cs_api, "load_session_case", lambda sid: None)
    monkeypatch.setattr(orch, "collect_stageA_logical_accounts", lambda sid: [])
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
    accounts = payload["accounts"]
    assert accounts["problem_accounts"] == []
    assert "negative_accounts" not in accounts
    assert "open_accounts_with_issues" not in accounts


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
    monkeypatch.setattr(config, "PROBLEM_DETECTION_ONLY", False)
    sample = {
        "account_id": "1",
        "bureau": "Experian",
        "primary_issue": "charge_off",
        "tier": "Tier1",
        "problem_reasons": ["reason"],
        "confidence": 0.9,
        "decision_source": "ai",
    }

    class DummyResult:
        def get(self, timeout=None):
            return {"problem_accounts": [sample]}

    monkeypatch.setattr(app_module, "run_full_pipeline", lambda sid: DummyResult())
    monkeypatch.setattr(app_module, "run_credit_repair_process", lambda *a, **k: None)
    monkeypatch.setattr(app_module, "set_session", lambda *a, **k: None)
    monkeypatch.setattr(app_module.cs_api, "load_session_case", lambda sid: None)
    monkeypatch.setattr(orch, "collect_stageA_logical_accounts", lambda sid: [sample])

    test_app = create_app()
    client = test_app.test_client()
    data = {
        "email": "a@example.com",
        "file": (io.BytesIO(b"%PDF-1.4"), "test.pdf"),
    }
    resp = client.post(
        "/api/start-process?legacy=1", data=data, content_type="multipart/form-data"
    )
    assert resp.status_code == 200
    payload_json = json.loads(resp.data)
    accounts = payload_json["accounts"]
    assert (
        accounts["problem_accounts"]
        == accounts["negative_accounts"]
        == accounts["open_accounts_with_issues"]
        == [sample]
    )


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
        "emitted_account" in r.message
        and "payment_statuses={'TransUnion': 'Collection/Chargeoff'}" in r.message
        for r in caplog.records
    )


def test_emitted_account_logs_co_marker(monkeypatch, caplog):
    monkeypatch.setattr(config, "PROBLEM_DETECTION_ONLY", False)
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
    assert payload.disputes[0].primary_issue == "charge_off"
