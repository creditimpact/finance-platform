import json
import os
import re

import pytest

from backend.core.case_store.telemetry import set_emitter
from backend.core.logic.report_analysis import candidate_logger as cl


EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"(?:\+?1[\s.-]?)?(?:\(?\d{3}\)?[\s.-]?)\d{3}[\s.-]?\d{4}")
SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b|\b\d{9}\b")
LONG_DIGITS_RE = re.compile(r"\b\d{8,}\b")
ADDRESS_RE = re.compile(
    r"\b(?:street|st\.|ave|road|rd\.|blvd|apt|suite)\b", re.IGNORECASE
)


def _configure(monkeypatch, tmp_path, fmt="jsonl"):
    monkeypatch.setattr(cl, "CASESTORE_DIR", tmp_path.as_posix())
    monkeypatch.setattr(cl, "ENABLE_CANDIDATE_TOKEN_LOGGER", True)
    monkeypatch.setattr(cl, "CANDIDATE_LOG_FORMAT", fmt)


def _collect(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _has_pii(text: str) -> bool:
    return any(
        regex.search(text)
        for regex in [EMAIL_RE, PHONE_RE, SSN_RE, LONG_DIGITS_RE, ADDRESS_RE]
    )


def test_jsonl_logging_sanitizes_and_emits(tmp_path, monkeypatch):
    _configure(monkeypatch, tmp_path, fmt="jsonl")
    events = []
    set_emitter(lambda e, f: events.append((e, f)))

    session_id = "sess1"
    acc_id = "acc_001_Equifax"
    bureau = "Equifax"
    fields = {
        "payment_status": "120D late",
        "account_status": "Closed",
        "past_due_amount": 125.0,
        "two_year_payment_history": ["OK", "30", "60"],
        "creditor_remarks": "Contact me at john.doe@x.com, (415) 555-1212",
    }
    decision = {
        "decision_source": "rules",
        "primary_issue": "unknown",
        "tier": "none",
        "confidence": 0.0,
        "problem_reasons": ["past_due_amount: 125.00"],
    }

    cl.log_stageA_candidates(session_id, acc_id, bureau, "pre", fields, {}, meta={"source": "stageA"})
    cl.log_stageA_candidates(session_id, acc_id, bureau, "post", fields, decision, meta={"source": "stageA"})
    # duplicate write to ensure jsonl allows duplicates
    cl.log_stageA_candidates(session_id, acc_id, bureau, "post", fields, decision, meta={"source": "stageA"})

    path = cl.candidate_tokens_path(session_id)
    assert os.path.exists(path)
    rows = _collect(path)
    assert len(rows) == 3
    for row in rows:
        for key in [
            "session_id",
            "account_id",
            "bureau",
            "phase",
            "timestamp",
            "fields",
            "decision",
        ]:
            assert key in row
        serialized = json.dumps(row)
        assert not _has_pii(serialized)

    writes = [e for e in events if e[0] == "candidate_tokens_write"]
    assert len(writes) == 3
    assert all(w[1]["bytes_written"] > 0 for w in writes)


def test_json_array_mode_and_idempotency(tmp_path, monkeypatch):
    _configure(monkeypatch, tmp_path, fmt="json")
    session_id = "sess2"
    acc_id = "acc_002_Equifax"
    bureau = "Equifax"
    fields = {"payment_status": "OK"}
    cl.log_stageA_candidates(session_id, acc_id, bureau, "post", fields, {})
    cl.log_stageA_candidates(session_id, acc_id, bureau, "post", fields, {})

    path = cl.candidate_tokens_path(session_id)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert isinstance(data, list)
    assert len(data) == 2
    # Ensure atomic replace left no temporary files
    prefix = f"{session_id}.candidate_tokens"
    files = [p.name for p in tmp_path.glob(f"{prefix}*")]
    assert files == [os.path.basename(path)]


def test_logger_emits_error_on_io(tmp_path, monkeypatch):
    _configure(monkeypatch, tmp_path, fmt="jsonl")
    events = []
    set_emitter(lambda e, f: events.append((e, f)))
    monkeypatch.setattr(cl, "CASESTORE_DIR", "/proc")
    with pytest.raises(Exception):
        cl.log_stageA_candidates(
            "sess", "acc_003_Equifax", "Equifax", "pre", {}, {},
        )
    errors = [e for e in events if e[0] == "candidate_tokens_error"]
    assert errors

