import json
from datetime import datetime

import pytest

from backend.core.case_store import (
    CaseStoreError,
    ReportMeta,
    SessionCase,
    Summary,
    VALIDATION_FAILED,
    load_session_case_json,
)

EXAMPLE_JSON = {
    "session_id": "7a0f4d7e-1f3a-4b0e-9e8f-88f2c4b1fd00",
    "created_at": "2025-08-27T12:00:00Z",
    "report_meta": {
        "credit_report_date": "2025-08-01",
        "personal_information": {
            "name": "REDACTED",
            "dob": "1989-**-**",
            "current_address": "REDACTED",
        },
        "public_information": [],
        "inquiries": [],
        "raw_source": {"vendor": "SmartCredit", "version": None, "doc_fingerprint": "abc123"},
    },
    "summary": {
        "total_accounts": 12,
        "open_accounts": 9,
        "closed_accounts": 3,
        "delinquent": 2,
        "derogatory": 1,
        "balances": 21500,
        "payments": None,
        "public_records": 0,
        "inquiries_2y": 2,
    },
    "accounts": {
        "acc_001_Equifax": {
            "bureau": "Equifax",
            "fields": {
                "account_number": "****1234",
                "date_opened": "2019-03-10",
                "balance_owed": 5000,
                "credit_limit": 6000,
                "payment_status": "120D late",
                "account_rating": "Derogatory",
            },
            "artifacts": {
                "stageA_detection": {
                    "primary_issue": "unknown",
                    "issue_types": [],
                    "problem_reasons": [],
                    "confidence": 0.0,
                    "tier": "none",
                    "decision_source": "rules",
                    "timestamp": "2025-08-27T12:01:20Z",
                }
            },
            "tags": {},
        }
    },
}


def test_round_trip_example():
    data = json.dumps(EXAMPLE_JSON)
    case = load_session_case_json(data)
    dumped = case.model_dump_json()
    again = load_session_case_json(dumped)
    assert again.session_id == EXAMPLE_JSON["session_id"]


def test_missing_summary_defaults():
    payload = dict(EXAMPLE_JSON)
    payload.pop("summary")
    data = json.dumps(payload)
    case = load_session_case_json(data)
    assert case.summary.total_accounts == 0


def test_missing_accounts_raises():
    payload = dict(EXAMPLE_JSON)
    payload.pop("accounts")
    data = json.dumps(payload)
    with pytest.raises(CaseStoreError) as exc:
        load_session_case_json(data)
    assert exc.value.code == VALIDATION_FAILED


def test_enum_and_invalid_value():
    payload = dict(EXAMPLE_JSON)
    acc = payload["accounts"]["acc_001_Equifax"]
    acc["bureau"] = "TransUnion"
    data = json.dumps(payload)
    case = load_session_case_json(data)
    assert case.accounts["acc_001_Equifax"].bureau.value == "TransUnion"

    acc["bureau"] = "BadBureau"
    data = json.dumps(payload)
    with pytest.raises(CaseStoreError):
        load_session_case_json(data)


def test_wrong_type_validation_error():
    payload = dict(EXAMPLE_JSON)
    payload["summary"]["balances"] = "oops"
    data = json.dumps(payload)
    with pytest.raises(CaseStoreError):
        load_session_case_json(data)


def test_optionals_accept_none():
    case = SessionCase(
        session_id="1",
        accounts={},
        summary=Summary(),
        report_meta=ReportMeta(),
    )
    assert case.summary.balances is None
