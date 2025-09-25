from __future__ import annotations

import json

from backend.core.logic.report_analysis.ai_adjudicator import persist_ai_decision


def test_persist_ai_decision_idempotent(tmp_path):
    sid = "case-xyz"
    runs_root = tmp_path

    response = {
        "decision": "merge",
        "confidence": 0.67,
        "reason": "matching account number",
        "reasons": ["matching account number", "aligned balances"],
        "flags": {"account_match": True, "debt_match": True},
    }

    persist_ai_decision(sid, runs_root, 101, 202, response)

    base = runs_root / sid / "cases" / "accounts"
    tags_a_path = base / "101" / "tags.json"
    tags_b_path = base / "202" / "tags.json"

    first_tags_a = json.loads(tags_a_path.read_text(encoding="utf-8"))
    first_tags_b = json.loads(tags_b_path.read_text(encoding="utf-8"))

    # Re-running with the same payload should replace instead of appending.
    persist_ai_decision(sid, runs_root, 101, 202, response)

    second_tags_a = json.loads(tags_a_path.read_text(encoding="utf-8"))
    second_tags_b = json.loads(tags_b_path.read_text(encoding="utf-8"))

    assert len(first_tags_a) == len(second_tags_a) == 1
    assert len(first_tags_b) == len(second_tags_b) == 1

    expected = {
        "kind": "merge_result",
        "with": 202,
        "decision": "merge",
        "confidence": 0.67,
        "reason": "matching account number",
        "reasons": ["matching account number", "aligned balances"],
        "flags": {"account_match": True, "debt_match": True},
        "source": "ai_adjudicator",
    }
    expected_b = dict(expected)
    expected_b["with"] = 101
    assert second_tags_a[0] == expected
    assert second_tags_b[0] == expected_b


def test_persist_ai_decision_overwrites_with_new_result(tmp_path):
    sid = "case-abc"
    runs_root = tmp_path

    first_response = {
        "decision": "merge",
        "confidence": 0.9,
        "reason": "identical creditor",
        "reasons": ["identical creditor"],
        "flags": {"account_match": True, "debt_match": True},
    }

    # Initial write with one ordering of the pair.
    persist_ai_decision(sid, runs_root, 101, 303, first_response)

    second_response = {
        "decision": "different",
        "confidence": 0.2,
        "reason": "different payment history",
        "flags": {"account_match": False, "debt_match": False},
    }

    # Re-running with the reversed ordering should still update both artifacts.
    persist_ai_decision(sid, runs_root, 303, 101, second_response)

    base = runs_root / sid / "cases" / "accounts"
    expected_tag_a = {
        "kind": "merge_result",
        "with": 303,
        "decision": "different",
        "confidence": 0.2,
        "reason": "different payment history",
        "reasons": ["different payment history"],
        "flags": {"account_match": False, "debt_match": False},
        "source": "ai_adjudicator",
    }
    expected_tag_b = dict(expected_tag_a)
    expected_tag_b["with"] = 101

    tags_a_path = base / "101" / "tags.json"
    tags_b_path = base / "303" / "tags.json"

    tags_a = json.loads(tags_a_path.read_text(encoding="utf-8"))
    tags_b = json.loads(tags_b_path.read_text(encoding="utf-8"))

    assert tags_a == [expected_tag_a]
    assert tags_b == [expected_tag_b]
    assert not (base / "101" / "ai").exists()
    assert not (base / "303" / "ai").exists()

