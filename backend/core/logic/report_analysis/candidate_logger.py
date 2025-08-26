"""PII-safe logging for Stage A candidate tokens.

This module provides utilities to persist a minimal set of account fields for
offline analysis.  The logger is designed to be safe-by-default, append-only and
observable via telemetry.  Historical helpers ``CandidateTokenLogger`` and
``StageATraceLogger`` are retained for backwards compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass  # noqa: F401  (imported for parity with spec)
from datetime import datetime
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from backend.config import (
    CASESTORE_DIR,
    ENABLE_CANDIDATE_TOKEN_LOGGER,
    CANDIDATE_LOG_FORMAT,
)
from backend.core.case_store.telemetry import emit


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def candidate_tokens_path(session_id: str) -> str:
    """Return absolute candidate tokens path based on ``CANDIDATE_LOG_FORMAT``."""

    ext = "jsonl" if CANDIDATE_LOG_FORMAT == "jsonl" else "json"
    return os.path.join(CASESTORE_DIR, f"{session_id}.candidate_tokens.{ext}")


_ALLOWED_FIELDS = {
    "payment_status",
    "account_status",
    "creditor_remarks",
    "past_due_amount",
    "balance_owed",
    "credit_limit",
    "high_balance",
    "two_year_payment_history",
    "days_late_7y",
    "account_type",
    "creditor_type",
    "dispute_status",
}


# PII regexes ---------------------------------------------------------------

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"(?:\+?1[\s.-]?)?(?:\(?\d{3}\)?[\s.-]?)\d{3}[\s.-]?\d{4}")
SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b|\b\d{9}\b")
LONG_DIGITS_RE = re.compile(r"\b\d{8,}\b")
ADDRESS_RE = re.compile(
    r"\b(?:street|st\.|ave|road|rd\.|blvd|apt|suite)\b", re.IGNORECASE
)


def _sanitize_str(value: str) -> str:
    value = EMAIL_RE.sub("[redacted]", value)
    value = PHONE_RE.sub("[redacted]", value)
    value = SSN_RE.sub("[redacted]", value)

    def _mask_long_digits(match: re.Match[str]) -> str:
        digits = match.group()
        return "****" + digits[-4:]

    value = LONG_DIGITS_RE.sub(_mask_long_digits, value)
    if ADDRESS_RE.search(value):
        return "[redacted]"
    return value


def _sanitize_value(val: Any) -> Any:
    if isinstance(val, str):
        return _sanitize_str(val)
    if isinstance(val, list):
        return [_sanitize_value(v) for v in val]
    if isinstance(val, dict):
        return {k: _sanitize_value(v) for k, v in val.items()}
    return val


def sanitize_fields_for_tokens(fields: Dict[str, Any]) -> Dict[str, Any]:
    """Deep-copy & sanitize fields for logging."""

    cleaned: Dict[str, Any] = {}
    for name in _ALLOWED_FIELDS:
        if name in fields and fields[name] is not None:
            cleaned[name] = _sanitize_value(fields[name])
    return cleaned


def log_stageA_candidates(
    session_id: str,
    account_id: str,
    bureau: str,
    phase: str,
    fields: Dict[str, Any],
    decision: Dict[str, Any],
    meta: Optional[Dict[str, Any]] = None,
) -> None:
    """Write one record to the session's candidate tokens file."""

    if not ENABLE_CANDIDATE_TOKEN_LOGGER:
        return

    record = {
        "session_id": session_id,
        "account_id": account_id,
        "bureau": bureau,
        "phase": phase,
        "timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "fields": sanitize_fields_for_tokens(fields),
        "decision": decision,
    }
    if meta:
        record["meta"] = meta

    path = candidate_tokens_path(session_id)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = json.dumps(record, ensure_ascii=False)
    bytes_written = len(data.encode("utf-8"))

    try:
        if CANDIDATE_LOG_FORMAT == "jsonl":
            with open(path, "a", encoding="utf-8") as f:
                f.write(data + "\n")
                f.flush()
                os.fsync(f.fileno())
            bytes_written += 1  # newline
        else:
            records: List[Dict[str, Any]] = []
            if os.path.exists(path):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        loaded = json.load(f)
                        if isinstance(loaded, list):
                            records = loaded
                except Exception:
                    records = []
            records.append(record)
            fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(path), text=True)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(records, f, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, path)

        try:
            emit(
                "candidate_tokens_write",
                session_id=session_id,
                account_id=account_id,
                phase=phase,
                bytes_written=bytes_written,
            )
        except Exception:
            pass
    except Exception:
        try:
            emit(
                "candidate_tokens_error",
                session_id=session_id,
                account_id=account_id,
                phase=phase,
                error="IO_ERROR",
            )
        except Exception:
            pass
        raise


# ---------------------------------------------------------------------------
# Legacy helpers retained for backwards compatibility
# ---------------------------------------------------------------------------


_FIELDS = [
    "balance_owed",
    "account_rating",
    "account_description",
    "dispute_status",
    "creditor_type",
    "account_status",
    "payment_status",
    "creditor_remarks",
    "account_type",
    "credit_limit",
    "late_payments",
    "past_due_amount",
]


def _redact(value: str) -> str:
    """Mask digits in string values to avoid logging PII."""
    if value.isdigit():
        return value
    return re.sub(r"\d", "X", value)


class CandidateTokenLogger:
    """Accumulates raw field values and persists them to disk."""

    def __init__(self) -> None:
        self._tokens: Dict[str, Set[str]] = {name: set() for name in _FIELDS}

    def collect(self, account: Dict[str, Any]) -> None:
        for field in _FIELDS:
            val = account.get(field)
            if val is None or val == "":
                continue
            if field == "late_payments" and isinstance(val, dict):
                for bureau, buckets in val.items():
                    for days, count in (buckets or {}).items():
                        token = f"{bureau}:{days}:{count}"
                        self._tokens[field].add(token)
            elif isinstance(val, dict):
                for v in val.values():
                    if v:
                        s = str(v)
                        self._tokens[field].add(_redact(s))
            else:
                s = str(val)
                if isinstance(val, (int, float)) or s.isdigit():
                    self._tokens[field].add(s)
                else:
                    self._tokens[field].add(_redact(s))

    def save(self, folder: Path) -> None:
        """Write collected tokens to ``folder/candidate_tokens.json``."""

        data = {k: sorted(v) for k, v in self._tokens.items() if v}
        if not data:
            return
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / "candidate_tokens.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)


class StageATraceLogger:
    """Append per-account Stage A decisions to a JSONL trace file."""

    def __init__(self, session_id: str, base_folder: Path | None = None) -> None:
        base = base_folder or Path("uploads")
        self.path = base / session_id / "stageA_trace.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, row: Dict[str, Any]) -> None:
        data = dict(row)
        data["ts"] = datetime.utcnow().isoformat() + "Z"
        with self.path.open("a", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
            f.write("\n")

