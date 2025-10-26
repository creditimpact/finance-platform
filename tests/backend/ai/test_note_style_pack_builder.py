"""Tests for the lightweight note_style pack builder."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.ai.note_style.pack_builder import PackBuilderError, build_pack
from backend.ai.note_style.prompt import build_base_system_prompt


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

    assert pack_payload["sid"] == sid
    assert pack_payload["account_id"] == account_id
    assert pack_payload["channel"] == "frontend_review"
    assert pack_payload["note_text"] == "Customer says the balance is wrong and wants help."

    note_metrics = pack_payload["note_metrics"]
    assert note_metrics == {"char_len": 50, "word_len": 9}

    account_payload = pack_payload["account_payload"]
    assert account_payload["meta"]["heading_guess"] == "Capital One Services"
    assert account_payload["meta"]["account_number_tail"] == "1234"
    assert account_payload["bureaus"]["transunion"]["balance_owed"] == "$100.00"
    assert account_payload["bureaus"]["equifax"]["payment_status"] == "Late 30 Days"
    assert account_payload["tags"] == tags_payload

    account_context = pack_payload["account_context"]
    meta_context = account_context.get("meta", {})
    assert meta_context.get("heading_guess") == "Capital One Services"
    tags_context = account_context.get("tags", {})
    assert tags_context.get("issues") == ["late_payment", "balance_dispute"]

    bureaus_summary = pack_payload["bureaus_summary"]
    assert bureaus_summary["majority_values"]["reported_creditor"] == "Capital One"
    assert bureaus_summary["majority_values"]["balance_owed"] == "100"
    assert "reported_creditor" in bureaus_summary["disagreements"]
    assert bureaus_summary["per_bureau"]["transunion"]["balance_owed"] == "100"
    assert bureaus_summary["per_bureau"]["experian"]["date_reported"] == "2024-03-10"

    pack_path = tmp_path / sid / "ai_packs" / "note_style" / "packs" / "acc_idx-007.jsonl"
    assert pack_path.is_file()
    lines = pack_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0]) == pack_payload

    debug_path = tmp_path / sid / "ai_packs" / "note_style" / "debug" / "idx-007.context.json"
    assert not debug_path.exists()

    messages = pack_payload["messages"]
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == build_base_system_prompt()
    assert messages[1]["role"] == "user"
    assert messages[1]["content"]["note_text"] == pack_payload["note_text"]
    assert messages[1]["content"]["context"] == account_payload


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
