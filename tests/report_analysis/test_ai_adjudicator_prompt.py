from __future__ import annotations

import json

from backend.core.logic.report_analysis.ai_adjudicator import build_prompt_from_pack


def test_build_prompt_from_pack_limits_context(monkeypatch):
    monkeypatch.setenv("AI_PACK_MAX_LINES_PER_SIDE", "9")

    pack = {
        "sid": "sample-sid",
        "pair": {"a": 11, "b": 16},
        "ids": {
            "account_number_a": "1234",
            "account_number_b": "5678",
            "account_number_a_normalized": "1234",
            "account_number_b_normalized": "5678",
            "account_number_a_last4": "1234",
            "account_number_b_last4": "5678",
        },
        "highlights": {
            "total": 81,
            "identity_score": 38,
            "debt_score": 22,
            "triggers": ["strong:balance_owed", "mid:dates"],
            "parts": {"balance_owed": 42, "account_number": 10},
            "matched_fields": {"balance_owed": True},
            "conflicts": ["amount_conflict:high_balance"],
            "acctnum_level": "exact_or_known_match",
            "ignored": "value",
        },
        "context": {
            "a": ["Creditor A", "Account # 1234", "Balance: 500", "Extra A"],
            "b": [
                "Creditor B",
                "Account # 5678",
                "Balance: 500",
                "Status: Open",
            ],
        },
        "limits": {"max_lines_per_side": 3},
    }

    prompt = build_prompt_from_pack(pack)

    assert set(prompt.keys()) == {"system", "user"}
    assert "expert credit tradeline merge adjudicator" in prompt["system"].lower()
    assert "decision" in prompt["system"]
    assert (
        "If only last four digits match but stems differ, never choose any same_account_*"
        in prompt["system"]
    )

    user_payload = json.loads(prompt["user"])
    assert user_payload["sid"] == "sample-sid"
    assert user_payload["pair"] == {"a": 11, "b": 16}
    assert user_payload["account_numbers"] == {"a": "1234", "b": "5678"}
    assert user_payload["account_numbers_normalized"] == {"a": "1234", "b": "5678"}
    assert user_payload["account_numbers_last4"] == {"a": "1234", "b": "5678"}
    assert user_payload["highlights"]["total"] == 81
    assert user_payload["highlights"]["identity_score"] == 38
    assert user_payload["highlights"]["debt_score"] == 22
    assert "ignored" not in user_payload["highlights"]
    assert len(user_payload["context"]["a"]) == 3
    assert user_payload["context"]["a"][-1] == "Balance: 500"
    assert len(user_payload["context"]["b"]) == 3
    assert user_payload["context"]["b"][-1] == "Balance: 500"


def test_build_prompt_from_pack_uses_pack_limit_when_lower(monkeypatch):
    monkeypatch.setenv("AI_PACK_MAX_LINES_PER_SIDE", "8")

    pack = {
        "sid": "other-sid",
        "pair": {"a": 1, "b": 2},
        "highlights": {},
        "context": {
            "a": ["A1", "A2", "A3", "A4"],
            "b": ["B1", "B2", "B3", "B4"],
        },
        "limits": {"max_lines_per_side": 2},
    }

    prompt = build_prompt_from_pack(pack)
    payload = json.loads(prompt["user"])

    assert payload["context"]["a"] == ["A1", "A2"]
    assert payload["context"]["b"] == ["B1", "B2"]
