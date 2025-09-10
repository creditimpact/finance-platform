import io
import json
import uuid

import pytest

import backend.config as config
from backend.api import app as app_module
from backend.api.app import create_app
from backend.core import orchestrators as orch
from backend.core.ai.models import AIAdjudicateResponse
from backend.core.case_store import api as cs_api, telemetry
from backend.core.logic.report_analysis import problem_detection as pd


class DummyResult:
    def get(self, timeout=None):
        return {}


def _setup_cross_bureau_case(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "CASESTORE_DIR", str(tmp_path))
    monkeypatch.setattr(config, "ENABLE_CASESTORE_STAGEA", True)
    monkeypatch.setattr(config, "ENABLE_AI_ADJUDICATOR", True)
    monkeypatch.setattr(config, "API_INCLUDE_DECISION_META", False)
    monkeypatch.setattr(app_module, "run_full_pipeline", lambda sid: DummyResult())
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

    cs_api.upsert_account_fields(
        session_id,
        "acc_exp",
        "Experian",
        dict(base, account_number="00001234", past_due_amount=0.0),
    )
    cs_api.set_tags(session_id, "acc_exp", person_id="p1")
    cs_api.append_artifact(
        session_id,
        "acc_exp",
        "stageA_detection",
        {
            "primary_issue": "collection",
            "tier": "Tier2",
            "confidence": 0.4,
            "problem_reasons": ["late"],
            "decision_source": "ai",
        },
    )

    cs_api.upsert_account_fields(
        session_id,
        "acc_tu",
        "TransUnion",
        dict(base, account_number="99991234", past_due_amount=0.0),
    )
    cs_api.set_tags(session_id, "acc_tu", person_id="p1")
    cs_api.append_artifact(
        session_id,
        "acc_tu",
        "stageA_detection",
        {
            "primary_issue": "collection",
            "tier": "Tier1",
            "confidence": 0.6,
            "problem_reasons": ["charge"],
            "decision_source": "ai",
        },
    )

    return session_id


def test_start_process_problem_accounts_filtered(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "CASESTORE_DIR", str(tmp_path))
    monkeypatch.setattr(config, "ENABLE_CASESTORE_STAGEA", True)
    monkeypatch.setattr(config, "ENABLE_AI_ADJUDICATOR", True)
    monkeypatch.setattr(config, "API_INCLUDE_DECISION_META", False)

    monkeypatch.setattr(app_module, "run_full_pipeline", lambda sid: DummyResult())
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
    cs_api.upsert_account_fields(
        session_id, "acc_ai", "Experian", dict(base, past_due_amount=0.0)
    )
    cs_api.upsert_account_fields(
        session_id, "acc_rules", "Experian", dict(base, past_due_amount=125.0)
    )
    cs_api.upsert_account_fields(
        session_id, "acc_clean", "Experian", dict(base, past_due_amount=0.0)
    )
    cs_api.upsert_account_fields(
        session_id,
        "acc_t4",
        "Experian",
        dict(base, past_due_amount=0.0, balance_owed=0.0),
    )

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
    resp = client.post(
        "/api/start-process", data=data, content_type="multipart/form-data"
    )
    assert resp.status_code == 200
    payload = json.loads(resp.data)
    accounts = payload["accounts"]["problem_accounts"]
    ids = {a["account_id"] for a in accounts}
    assert {"acc_ai", "acc_rules"} <= ids
    assert "acc_clean" not in ids

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


def test_start_process_cross_bureau_flag_off(monkeypatch, tmp_path):
    _setup_cross_bureau_case(monkeypatch, tmp_path)
    monkeypatch.setattr(config, "ENABLE_CROSS_BUREAU_RESOLUTION", False)

    test_app = create_app()
    client = test_app.test_client()
    data = {"email": "a@example.com", "file": (io.BytesIO(b"%PDF-1.4"), "test.pdf")}
    resp = client.post(
        "/api/start-process", data=data, content_type="multipart/form-data"
    )
    assert resp.status_code == 200
    payload = json.loads(resp.data)
    accounts = payload["accounts"]["problem_accounts"]
    assert len(accounts) == 2
    bureaus = {a["bureau"] for a in accounts}
    assert bureaus == {"Experian", "TransUnion"}


def test_start_process_cross_bureau_flag_on(monkeypatch, tmp_path):
    _setup_cross_bureau_case(monkeypatch, tmp_path)
    monkeypatch.setattr(config, "ENABLE_CROSS_BUREAU_RESOLUTION", True)
    monkeypatch.setattr(config, "API_AGGREGATION_ID_STRATEGY", "winner")
    monkeypatch.setattr(config, "API_INCLUDE_AGG_MEMBERS_META", False)
    events: list[tuple[str, dict]] = []
    telemetry.set_emitter(lambda e, f: events.append((e, f)))

    test_app = create_app()
    client = test_app.test_client()
    data = {"email": "a@example.com", "file": (io.BytesIO(b"%PDF-1.4"), "test.pdf")}
    resp = client.post(
        "/api/start-process", data=data, content_type="multipart/form-data"
    )
    telemetry.set_emitter(None)
    assert resp.status_code == 200
    payload = json.loads(resp.data)
    accounts = payload["accounts"]["problem_accounts"]
    assert len(accounts) == 1
    acc = accounts[0]
    assert acc["bureau"] == "TransUnion"
    assert acc["account_id"] == "acc_tu"
    assert acc["tier"] == "Tier1"
    assert acc["confidence"] == pytest.approx(0.6)
    assert set(acc["problem_reasons"]) == {"late", "charge"}
    assert acc["decision_source"] == "ai"
    assert "aggregation_meta" not in acc
    assert any(e == "stageA_cross_bureau_aggregated" for e, _ in events)


def test_start_process_cross_bureau_logical_strategy(monkeypatch, tmp_path):
    session_id = _setup_cross_bureau_case(monkeypatch, tmp_path)
    monkeypatch.setattr(config, "ENABLE_CROSS_BUREAU_RESOLUTION", True)
    monkeypatch.setattr(config, "API_AGGREGATION_ID_STRATEGY", "logical")
    monkeypatch.setattr(config, "API_INCLUDE_AGG_MEMBERS_META", True)
    events: list[tuple[str, dict]] = []
    telemetry.set_emitter(lambda e, f: events.append((e, f)))

    test_app = create_app()
    client = test_app.test_client()
    data = {"email": "a@example.com", "file": (io.BytesIO(b"%PDF-1.4"), "test.pdf")}
    resp = client.post(
        "/api/start-process", data=data, content_type="multipart/form-data"
    )
    telemetry.set_emitter(None)
    assert resp.status_code == 200
    payload = json.loads(resp.data)
    accounts = payload["accounts"]["problem_accounts"]
    assert len(accounts) == 1
    acc = accounts[0]
    case = cs_api.get_account_case(session_id, "acc_exp")
    expected_key = orch.compute_logical_account_key(case)
    assert acc["account_id"] == expected_key
    assert acc["bureau"] == "TransUnion"
    assert "aggregation_meta" in acc
    meta = acc["aggregation_meta"]
    assert meta["logical_account_id"] == expected_key
    members = {m["account_id"] for m in meta["members"]}
    assert members == {"acc_exp", "acc_tu"}
    assert any(e == "stageA_cross_bureau_aggregated" for e, _ in events)
