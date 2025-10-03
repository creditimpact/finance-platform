import json

import pytest

from backend.core.case_store import telemetry
from backend.core.logic.report_analysis import candidate_logger
from backend.core.logic.validation_field_sets import ALL_VALIDATION_FIELDS


@pytest.fixture
def capture_telemetry():
    events = []

    def _emit(event, fields):
        events.append((event, fields))

    telemetry.set_emitter(_emit)
    yield events
    telemetry.set_emitter(None)


@pytest.mark.parametrize("log_format", ["jsonl", "json"])
def test_mask_counts_mixed_pii(tmp_path, monkeypatch, capture_telemetry, log_format):
    monkeypatch.setattr(candidate_logger, "CASESTORE_DIR", str(tmp_path))
    monkeypatch.setattr(candidate_logger, "CANDIDATE_LOG_FORMAT", log_format)

    fields = {
        "creditor_remarks": "Contact at foo@example.com and bar@example.com",
        "payment_status": "SSN 123-45-6789",
        "account_status": "Call +1 555-123-4567",
        "creditor_type": "123 Main St.",
        "balance_owed": "123456789012",
        "high_balance": "987654321098",
    }

    candidate_logger.log_stageA_candidates(
        "sess1", "acct1", "bureauX", "pre", fields, decision={}
    )

    event, payload = capture_telemetry[0]
    assert event == "candidate_tokens_write"
    assert payload["records"] == 1
    assert payload["fields_masked_email"] == 2
    assert payload["fields_masked_phone"] == 1
    assert payload["fields_masked_ssn"] == 1
    assert payload["fields_masked_address"] == 1
    assert payload["fields_masked_account"] == 2
    assert payload["fields_masked_total"] == 7

    dumped = json.dumps(payload)
    for token in [
        "foo@example.com",
        "bar@example.com",
        "123-45-6789",
        "555-123-4567",
        "123 Main St",
        "123456789012",
        "987654321098",
    ]:
        assert token not in dumped


def test_mask_counts_idempotent(tmp_path, monkeypatch, capture_telemetry):
    monkeypatch.setattr(candidate_logger, "CASESTORE_DIR", str(tmp_path))
    monkeypatch.setattr(candidate_logger, "CANDIDATE_LOG_FORMAT", "jsonl")

    fields = {"creditor_remarks": "reach me at user@example.com"}

    candidate_logger.log_stageA_candidates(
        "sess2", "acct2", "bureauX", "pre", fields, decision={}
    )
    first_event = capture_telemetry[-1][1]
    assert first_event["fields_masked_total"] == 1

    sanitized, _ = candidate_logger.sanitize_fields_for_tokens(fields)
    candidate_logger.log_stageA_candidates(
        "sess2", "acct2", "bureauX", "pre", sanitized, decision={}
    )
    second_event = capture_telemetry[-1][1]
    assert second_event["fields_masked_total"] == 0


def test_candidate_logger_field_scope_matches_spec() -> None:
    assert set(candidate_logger._ALLOWED_FIELDS) == set(ALL_VALIDATION_FIELDS)
