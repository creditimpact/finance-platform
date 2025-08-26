import io
import json
import uuid

import pytest

import backend.config as config
from backend.api.app import create_app
from backend.api import app as app_module
from backend.core.ai.models import AIAdjudicateResponse
from backend.core.case_store import api as cs_api
from backend.core.logic.report_analysis import problem_detection as pd


class DummyResult:
    def get(self, timeout=None):
        return {}


class DummyTask:
    def delay(self, *a, **k):
        return DummyResult()


def test_start_process_problem_accounts_filtered(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "CASESTORE_DIR", str(tmp_path))
    monkeypatch.setattr(config, "ENABLE_CASESTORE_STAGEA", True)
    monkeypatch.setattr(config, "ENABLE_AI_ADJUDICATOR", True)

    monkeypatch.setattr(app_module, "extract_problematic_accounts", DummyTask())
    monkeypatch.setattr(app_module, "set_session", lambda *a, **k: None)

    session_id = "sess1"

    class DummyUUID:
        hex = "filehex"

        def __str__(self):
            return session_id

    monkeypatch.setattr(uuid, "uuid4", lambda: DummyUUID())

    case = cs_api.create_session_case(session_id)
    cs_api.save_session_case(case)
    base = {
        "balance_owed": 100.0,
        "credit_limit": 1000.0,
        "high_balance": 500.0,
        "payment_status": "",
        "account_status": "",
        "two_year_payment_history": "",
        "days_late_7y": "",
    }
    cs_api.upsert_account_fields(session_id, "acc_ai", "Experian", dict(base, past_due_amount=0.0))
    cs_api.upsert_account_fields(session_id, "acc_rules", "Experian", dict(base, past_due_amount=125.0))
    cs_api.upsert_account_fields(session_id, "acc_clean", "Experian", dict(base, past_due_amount=0.0))
    cs_api.upsert_account_fields(session_id, "acc_t4", "Experian", dict(base, past_due_amount=0.0))

    responses = [
        AIAdjudicateResponse(
            primary_issue="collection",
            tier="Tier1",
            confidence=0.9,
            problem_reasons=["ai_reason"],
            fields_used=["balance_owed"],
        ),
        AIAdjudicateResponse(
            primary_issue="collection",
            tier="Tier2",
            confidence=0.5,
            problem_reasons=["ai_reason"],
            fields_used=["balance_owed"],
        ),
        None,
        AIAdjudicateResponse(
            primary_issue="collection",
            tier="Tier4",
            confidence=0.95,
            problem_reasons=["ai_reason"],
            fields_used=["balance_owed"],
        ),
    ]

    def fake_call(session, req):
        return responses.pop(0)

    monkeypatch.setattr(pd, "call_adjudicator", fake_call)

    pd.run_stage_a(session_id, [])

    test_app = create_app()
    client = test_app.test_client()
    data = {"email": "a@example.com", "file": (io.BytesIO(b"%PDF-1.4"), "test.pdf")}
    resp = client.post("/api/start-process", data=data, content_type="multipart/form-data")
    assert resp.status_code == 200
    payload = json.loads(resp.data)
    accounts = payload["accounts"]["problem_accounts"]
    ids = {a["account_id"] for a in accounts}
    assert ids == {"acc_ai", "acc_rules"}

    for acc in accounts:
        expected = {
            "account_id",
            "bureau",
            "primary_issue",
            "tier",
            "problem_reasons",
            "confidence",
            "decision_source",
        }
        assert expected <= acc.keys()
        assert acc.keys() <= expected | {"fields_used"}

    ai_acc = next(a for a in accounts if a["account_id"] == "acc_ai")
    assert ai_acc["decision_source"] == "ai"
    assert ai_acc["tier"] == "Tier1"

    rules_acc = next(a for a in accounts if a["account_id"] == "acc_rules")
    assert rules_acc["decision_source"] == "rules"
    assert rules_acc["tier"] == "none"
    assert rules_acc["primary_issue"] == "unknown"

