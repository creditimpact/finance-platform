"""Tests for the lightweight note_style pack builder."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.ai.note_style.pack_builder import PackBuilderError, build_pack


EXPECTED_SYSTEM_PROMPT = (
    "You analyse customer notes and respond with structured JSON. Return exactly one JSON "
    "object using this schema: {\"tone\": string, \"context_hints\": {\"timeframe\": {\"month\": "
    "string|null, \"relative\": string|null}, \"topic\": string, \"entities\": {\"creditor\": "
    "string|null, \"amount\": number|null}}, \"emphasis\": [string], \"confidence\": number, "
    "\"risk_flags\": [string]}. Never include explanations or additional keys."
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def test_build_pack_collects_context_and_writes_jsonl(tmp_path: Path) -> None:
    sid = "SID-123"
    account_id = "idx-007"

    response_payload = {
        "answers": {"explanation": "Customer says the balance is wrong and wants help."}
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
            "date_reported": "03/10/2024",
            "date_of_last_activity": "02/05/2024",
            "closed_date": None,
            "last_verified": "2024/03/01",
            "balance_owed": "$100.00",
            "high_balance": "$250",
            "past_due_amount": "$0",
            "account_number_display": "****1234",
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
            "last_verified": "03/01/2024",
            "balance_owed": "$100",
            "high_balance": "$250",
            "past_due_amount": "$0",
            "account_number_display": "****1234",
        },
        "equifax": {
            "reported_creditor": "Capital One",
            "account_type": "Credit Card",
            "account_status": "Closed",
            "payment_status": "Late 30 Days",
            "creditor_type": "Bank",
            "date_opened": "2020-01-15",
            "date_reported": "2024.03.10",
            "date_of_last_activity": "2024-02-05",
            "closed_date": "12/31/2023",
            "last_verified": "2024-03-01T00:00:00",
            "balance_owed": "$120",
            "high_balance": "$300",
            "past_due_amount": "$20",
            "account_number_display": "****1234",
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

    pack_payload = build_pack(sid, account_id, runs_root=tmp_path)

    assert set(pack_payload) == {
        "meta_name",
        "primary_issue_tag",
        "bureau_data",
        "note_text",
        "messages",
    }
    assert pack_payload["meta_name"] == "Capital One Services"
    assert pack_payload["primary_issue_tag"] == "late_payment"
    assert (
        pack_payload["note_text"]
        == "Customer says the balance is wrong and wants help."
    )

    bureau_data = pack_payload["bureau_data"]
    assert bureau_data["account_type"] == "Credit Card"
    assert bureau_data["account_status"] == "Open"

    pack_path = tmp_path / sid / "ai_packs" / "note_style" / "packs" / "acc_idx-007.jsonl"
    assert pack_path.is_file()
    lines = pack_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0]) == pack_payload

    debug_path = tmp_path / sid / "ai_packs" / "note_style" / "debug" / "idx-007.context.json"
    assert not debug_path.exists()

    messages = pack_payload["messages"]
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == EXPECTED_SYSTEM_PROMPT
    assert messages[1]["role"] == "user"
    user_content = messages[1]["content"]
    assert user_content == {
        "meta_name": "Capital One Services",
        "primary_issue_tag": "late_payment",
        "bureau_data": bureau_data,
        "note_text": "Customer says the balance is wrong and wants help.",
    }


def test_build_pack_requires_existing_artifacts(tmp_path: Path) -> None:
    sid = "SID-404"
    account_id = "idx-999"

    response_path = (
        tmp_path
        / sid
        / "frontend"
        / "review"
        / "responses"
        / f"{account_id}.result.json"
    )
    _write_json(response_path, {"answers": {"explanation": "Missing account info"}})

    with pytest.raises(PackBuilderError):
        build_pack(sid, account_id, runs_root=tmp_path)
