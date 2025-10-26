from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.ai.note_style.pack_builder import PackBuilderError, build_pack
from backend.note_style.writer import write_failure_dump, write_result


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _prepare_pack(tmp_path: Path, sid: str, account_id: str) -> dict[str, object]:
    response_payload = {
        "answers": {"explanation": "Customer believes the balance is incorrect."}
    }
    response_path = (
        tmp_path
        / sid
        / "frontend"
        / "review"
        / "responses"
        / f"{account_id}.result.json"
    )
    _write_json(response_path, response_payload)

    account_dir = tmp_path / sid / "cases" / "accounts" / "7"
    bureaus_payload = {
        "transunion": {
            "reported_creditor": "Capital One",
            "account_type": "Credit Card",
            "account_status": "Open",
            "payment_status": "Current",
            "creditor_type": "Bank",
            "date_opened": "2020-01-15",
            "date_reported": "2024-03-10",
            "date_of_last_activity": "2024-02-05",
            "closed_date": None,
            "last_verified": "2024-03-01",
            "balance_owed": "$100",
            "high_balance": "$250",
            "past_due_amount": "$0",
        },
        "experian": {
            "reported_creditor": "Capital One Bank",
            "account_type": "Credit Card",
            "account_status": "Open",
            "payment_status": "Current",
            "creditor_type": "Bank",
            "date_opened": "2020-01-15",
            "date_reported": "2024-03-10",
            "date_of_last_activity": "2024-02-05",
            "closed_date": None,
            "last_verified": "2024-03-01",
            "balance_owed": "$100",
            "high_balance": "$250",
            "past_due_amount": "$0",
        },
        "equifax": {
            "reported_creditor": "Capital One",
            "account_type": "Credit Card",
            "account_status": "Closed",
            "payment_status": "Late 30 Days",
            "creditor_type": "Bank",
            "date_opened": "2020-01-15",
            "date_reported": "2024-03-10",
            "date_of_last_activity": "2024-02-05",
            "closed_date": "2023-12-31",
            "last_verified": "2024-03-01",
            "balance_owed": "$120",
            "high_balance": "$300",
            "past_due_amount": "$20",
        },
    }
    tags_payload = [
        {"kind": "issue", "type": "late_payment"},
        {"kind": "issue", "type": "balance_dispute"},
    ]
    meta_payload = {
        "heading_guess": "Capital One Services",
        "account_number_tail": "1234",
    }

    _write_json(account_dir / "bureaus.json", bureaus_payload)
    _write_json(account_dir / "tags.json", tags_payload)
    _write_json(account_dir / "meta.json", meta_payload)

    return build_pack(sid, account_id, runs_root=tmp_path)


def test_write_result_writes_minimal_payload(tmp_path: Path) -> None:
    sid = "SID-555"
    account_id = "idx-007"

    pack_payload = _prepare_pack(tmp_path, sid, account_id)

    analysis_payload = {
        "tone": "Supportive",
        "context_hints": {
            "timeframe": {"month": None, "relative": None},
            "topic": "payment_dispute",
            "entities": {"creditor": "Capital One", "amount": None},
        },
        "emphasis": ["Paid Already", "Custom"],
        "confidence": 0.7,
        "risk_flags": [],
    }

    result_path = write_result(
        sid,
        account_id,
        analysis_payload,
        runs_root=tmp_path,
        pack_payload=pack_payload,
    )

    expected_path = (
        tmp_path
        / sid
        / "ai_packs"
        / "note_style"
        / "results"
        / "acc_idx-007.result.jsonl"
    )
    assert result_path == expected_path
    assert result_path.is_file()

    lines = [line.strip() for line in result_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 1
    stored_payload = json.loads(lines[0])

    assert set(stored_payload.keys()) == {"sid", "account_id", "analysis", "note_metrics"}
    assert stored_payload["sid"] == sid
    assert stored_payload["account_id"] == account_id

    analysis = stored_payload["analysis"]
    assert analysis["tone"] == "Supportive"
    assert analysis["context_hints"]["topic"] == "payment_dispute"
    assert analysis["emphasis"] == ["paid already", "custom"]
    assert analysis["confidence"] == pytest.approx(0.7)
    assert analysis["risk_flags"] == []

    metrics = stored_payload["note_metrics"]
    assert metrics == {
        "char_len": len(pack_payload["context"]["note_text"]),
        "word_len": len(pack_payload["context"]["note_text"].split()),
    }

    # Ensure we did not leak contextual payloads into the result file.
    assert "account_context" not in stored_payload
    assert "bureaus_summary" not in stored_payload


def test_write_failure_dump_records_raw_payload(tmp_path: Path) -> None:
    sid = "SID-556"
    account_id = "idx-200"

    raw_payload = {"error": "schema_mismatch", "received": ["not", "json"]}

    raw_path = write_failure_dump(sid, account_id, raw_payload, runs_root=tmp_path)

    expected_raw = (
        tmp_path
        / sid
        / "ai_packs"
        / "note_style"
        / "results_raw"
        / "acc_idx-200.raw.txt"
    )

    assert raw_path == expected_raw
    assert raw_path.is_file()
    raw_text = raw_path.read_text(encoding="utf-8")
    assert "schema_mismatch" in raw_text
    assert raw_text.endswith("\n")

    result_path = (
        tmp_path
        / sid
        / "ai_packs"
        / "note_style"
        / "results"
        / "acc_idx-200.result.jsonl"
    )
    assert not result_path.exists()


def test_write_result_requires_metrics(tmp_path: Path) -> None:
    sid = "SID-557"
    account_id = "idx-300"

    with pytest.raises(PackBuilderError):
        build_pack(sid, account_id, runs_root=tmp_path)

    analysis_payload = {
        "tone": "Supportive",
        "context_hints": {
            "timeframe": {"month": None, "relative": None},
            "topic": "payment_dispute",
            "entities": {"creditor": "Capital One", "amount": None},
        },
        "emphasis": ["Paid Already"],
        "confidence": 0.8,
        "risk_flags": [],
    }

    with pytest.raises(ValueError):
        write_result(sid, account_id, analysis_payload, runs_root=tmp_path)
