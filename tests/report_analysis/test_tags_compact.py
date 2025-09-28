from __future__ import annotations

import json
from pathlib import Path

from backend.core.io.tags import read_tags
from backend.core.logic.report_analysis.tags_compact import compact_tags_for_account


def _account_dir(tmp_path: Path) -> Path:
    account_dir = tmp_path / "cases" / "accounts" / "7"
    account_dir.mkdir(parents=True, exist_ok=True)
    return account_dir


def _write_tags(account_dir: Path, payload: list[dict[str, object]]) -> None:
    tag_path = account_dir / "tags.json"
    tag_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_summary(account_dir: Path) -> dict[str, object]:
    summary_path = account_dir / "summary.json"
    if not summary_path.exists():
        return {}
    return json.loads(summary_path.read_text(encoding="utf-8"))


def test_compact_tags_moves_verbose_data_to_summary(tmp_path: Path) -> None:
    account_dir = _account_dir(tmp_path)
    summary_path = account_dir / "summary.json"
    summary_path.write_text(
        json.dumps({"existing": True, "merge_explanations": [{"kind": "merge_best", "with": 99}]}, indent=2),
        encoding="utf-8",
    )

    _write_tags(
        account_dir,
        [
            {
                "kind": "issue",
                "type": "collection",
                "details": {"problem_reasons": ["reason-a"]},
            },
            {
                "kind": "merge_best",
                "with": "16",
                "decision": "ai",
                "total": 59,
                "parts": {"balance_owed": 31},
                "aux": {
                    "acctnum_level": "none",
                    "matched_fields": {"balance_owed": True},
                    "by_field_pairs": {},
                },
                "reasons": ["strong:balance_owed", "mid", "total"],
                "conflicts": ["amount_conflict:high_balance"],
                "source": "merge_scorer",
            },
            {
                "kind": "ai_decision",
                "with": 16,
                "decision": "different",
                "reason": "Different account numbers and conflicting last payment dates; same debt indicated.",
                "raw_response": {
                    "decision": "different",
                    "reason": "Different account numbers and conflicting last payment dates; same debt indicated.",
                },
                "at": "2025-09-21T22:47:26Z",
            },
            {
                "kind": "same_debt_pair",
                "with": 16,
                "at": "2025-09-21T22:47:26Z",
            },
        ],
    )

    compact_tags_for_account(account_dir)

    tags_after = read_tags(account_dir)
    assert tags_after == [
        {"kind": "issue", "type": "collection"},
        {"kind": "merge_best", "with": 16, "decision": "ai"},
        {
            "kind": "ai_decision",
            "with": 16,
            "decision": "different",
            "at": "2025-09-21T22:47:26Z",
        },
        {"kind": "same_debt_pair", "with": 16, "at": "2025-09-21T22:47:26Z"},
        {
            "kind": "ai_resolution",
            "with": 16,
            "decision": "different",
            "flags": {},
            "reason": "Different account numbers and conflicting last payment dates; same debt indicated.",
        },
    ]

    summary_after = _read_summary(account_dir)
    assert summary_after["existing"] is True

    merge_entries = summary_after["merge_explanations"]
    assert isinstance(merge_entries, list)
    assert {entry.get("with") for entry in merge_entries} == {16, 99}
    merge_entry = next(entry for entry in merge_entries if entry.get("with") == 16)
    assert merge_entry["kind"] == "merge_best"
    assert merge_entry["total"] == 59
    assert merge_entry["parts"] == {"balance_owed": 31}
    assert merge_entry["conflicts"] == ["amount_conflict:high_balance"]
    assert merge_entry["matched_fields"] == {"balance_owed": True}
    aux_payload = merge_entry.get("aux", {}) if isinstance(merge_entry.get("aux"), dict) else {}
    acct_level = merge_entry.get("acctnum_level", aux_payload.get("acctnum_level"))
    assert acct_level == "none"

    ai_entries = summary_after["ai_explanations"]
    assert isinstance(ai_entries, list)
    assert {entry.get("kind") for entry in ai_entries} == {
        "ai_decision",
        "ai_resolution",
        "same_debt_pair",
    }
    ai_decision_entry = next(entry for entry in ai_entries if entry.get("kind") == "ai_decision")
    assert ai_decision_entry["with"] == 16
    assert ai_decision_entry["normalized"] is False
    assert ai_decision_entry["decision"] == "different"
    assert ai_decision_entry["reason"].startswith("Different account numbers")
    assert "raw_response" in ai_decision_entry
    ai_resolution_entry = next(
        entry for entry in ai_entries if entry.get("kind") == "ai_resolution"
    )
    assert ai_resolution_entry["with"] == 16
    assert ai_resolution_entry["normalized"] is False
    assert ai_resolution_entry["decision"] == "different"
    assert ai_resolution_entry["reason"].startswith("Different account numbers")
    same_debt_entry = next(entry for entry in ai_entries if entry.get("kind") == "same_debt_pair")
    assert same_debt_entry["reason"].startswith("Different account numbers")

    summary_snapshot = json.loads(summary_path.read_text(encoding="utf-8"))
    tags_snapshot = read_tags(account_dir)

    compact_tags_for_account(account_dir)

    assert read_tags(account_dir) == tags_snapshot
    assert json.loads(summary_path.read_text(encoding="utf-8")) == summary_snapshot

