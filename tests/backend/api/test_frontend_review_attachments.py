from __future__ import annotations

import json
from pathlib import Path
import sys
import types

if "requests" not in sys.modules:
    requests_stub = types.ModuleType("requests")

    class _StubSession:
        def get(self, *_args: object, **_kwargs: object) -> None:  # pragma: no cover - stub
            raise RuntimeError("requests stub invoked")

        def close(self) -> None:  # pragma: no cover - stub
            pass

    class _StubRequestException(Exception):
        pass

    requests_stub.Session = _StubSession
    requests_stub.RequestException = _StubRequestException
    sys.modules["requests"] = requests_stub

from backend.api.app import (
    _build_account_attachments_summary,
    _build_field_snapshots,
    _sanitize_attachment_record,
)


def test_build_field_snapshots_adds_default_code_when_inconsistent() -> None:
    summary_payload = {
        "validation_requirements": {
            "field_consistency": {"balance_owed": {"consensus": "split"}},
            "findings": [],
        }
    }
    bureaus_payload = {
        "transunion": {"balance_owed": "$100"},
        "experian": {"balance_owed": "$200"},
        "equifax": {"balance_owed": "$300"},
    }

    snapshots = _build_field_snapshots(
        summary_payload,
        bureaus_payload,
        ["balance_owed"],
    )

    assert snapshots["balance_owed"]["consistent"] is False
    assert snapshots["balance_owed"]["c_codes"] == ["INCONSISTENT_BUREAUS"]


def test_sanitize_attachment_record_normalizes_fields() -> None:
    entry = {
        "id": "att_123",
        "claim_type": "paid_in_full",
        "filename": "statement.pdf",
        "stored_path": "runs/demo/frontend/review/uploads/idx-001/file.pdf",
        "mime": "application/pdf",
        "size": 1024,
        "sha1": "abc123",
        "uploaded_at": "2024-01-01T00:00:00Z",
        "hot_fields": ["balance_owed", 42, ""],
        "field_snapshots": {
            "balance_owed": {
                "by_bureau": {
                    "transunion": "$100",
                    "experian": "$200",
                    "equifax": "$300",
                },
                "consistent": False,
                "c_codes": ["C01"],
            }
        },
        "doc_id": "runs/demo/frontend/review/uploads/idx-001/file.pdf",
        "ignored": "value",
    }

    sanitized = _sanitize_attachment_record(entry)

    assert sanitized["hot_fields"] == ["balance_owed"]
    assert sanitized["field_snapshots"]["balance_owed"]["c_codes"] == ["C01"]
    assert "ignored" not in sanitized


def test_build_account_attachments_summary_handles_missing(tmp_path: Path) -> None:
    summary = _build_account_attachments_summary(tmp_path, "idx-001", sid="sid123")

    assert summary == {
        "sid": "sid123",
        "account_id": "idx-001",
        "attachments": [],
        "hot_fields_union": [],
    }


def test_build_account_attachments_summary_reads_existing(tmp_path: Path) -> None:
    payload = {
        "attachments": [
            {
                "id": "att_1",
                "claim_type": "paid_in_full",
                "filename": "statement.pdf",
                "stored_path": "runs/demo/frontend/review/uploads/idx-001/file.pdf",
                "mime": "application/pdf",
                "size": 2048,
                "sha1": "abc",
                "uploaded_at": "2024-01-01T00:00:00Z",
                "hot_fields": ["balance_owed", "payment_status"],
                "field_snapshots": {
                    "balance_owed": {
                        "by_bureau": {
                            "transunion": "$100",
                            "experian": "$100",
                            "equifax": "$100",
                        },
                        "consistent": True,
                        "c_codes": [],
                    }
                },
            }
        ]
    }

    summary_path = tmp_path / "idx-001.summary.json"
    summary_path.write_text(json.dumps(payload), encoding="utf-8")

    summary = _build_account_attachments_summary(tmp_path, "idx-001", sid="sid123")

    assert summary["attachments"][0]["id"] == "att_1"
    assert summary["hot_fields_union"] == ["balance_owed", "payment_status"]
