from __future__ import annotations

import json

from backend.core.logic.report_analysis.ai_adjudicator import persist_ai_decision


def test_persist_ai_decision_idempotent(tmp_path):
    sid = "case-xyz"
    runs_root = tmp_path

    response = {
        "decision": "merge",
        "confidence": 0.67,
        "reasons": ["matching account number", "aligned balances"],
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

    assert second_tags_a[0] == {
        "kind": "merge_result",
        "with": 202,
        "decision": "merge",
        "confidence": 0.67,
        "reasons": ["matching account number", "aligned balances"],
        "source": "ai_adjudicator",
    }
    assert second_tags_b[0] == {
        "kind": "merge_result",
        "with": 101,
        "decision": "merge",
        "confidence": 0.67,
        "reasons": ["matching account number", "aligned balances"],
        "source": "ai_adjudicator",
    }

