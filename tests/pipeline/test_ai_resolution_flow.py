"""Minimal end-to-end tests for the AI resolution pipeline."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from backend.core.ai.paths import get_merge_paths
from scripts import send_ai_merge_packs


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


@pytest.mark.parametrize(
    "flags,expected_decision,expected_pair",
    [
        (
            {"account_match": True, "debt_match": True},
            "same_account_same_debt",
            "same_account_pair",
        ),
        (
            {"account_match": "true", "debt_match": "FALSE"},
            "same_account_diff_debt",
            "same_account_pair",
        ),
        (
            {"account_match": "false", "debt_match": "true"},
            "same_debt_diff_account",
            "same_debt_pair",
        ),
        (
            {"account_match": False, "debt_match": False},
            "different",
            None,
        ),
    ],
)
def test_send_packs_normalizes_decisions(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    flags: dict[str, bool | str],
    expected_decision: str,
    expected_pair: str | None,
) -> None:
    runs_root = tmp_path / "runs"
    sid = f"flow-{expected_decision}"
    merge_paths = get_merge_paths(runs_root, sid, create=True)
    packs_dir = merge_paths.packs_dir

    pack_filename = "pair_001_002.jsonl"
    pack_payload = {
        "messages": [
            {"role": "system", "content": "instructions"},
            {"role": "user", "content": "Account pair"},
        ]
    }
    (packs_dir / pack_filename).write_text(
        json.dumps(pack_payload, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    index_payload = {
        "sid": sid,
        "pairs": [
            {
                "a": 1,
                "b": 2,
                "pack_file": pack_filename,
                "lines_a": 0,
                "lines_b": 0,
                "score_total": 0,
            }
        ],
    }
    _write_json(merge_paths.index_file, index_payload)

    accounts_root = runs_root / sid / "cases" / "accounts"
    _write_json(accounts_root / "1" / "tags.json", [])
    _write_json(accounts_root / "2" / "tags.json", [])

    monkeypatch.setenv("ENABLE_AI_ADJUDICATOR", "1")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("RUNS_ROOT", str(runs_root))

    reason = f"Stub decision for {expected_decision}"
    timestamp = "2024-07-01T00:00:00Z"

    def _fake_decide(pack: dict[str, Any], *, timeout: float) -> dict[str, Any]:
        messages = pack.get("messages")
        expected_messages = pack_payload.get("messages")
        assert isinstance(messages, list)
        assert isinstance(expected_messages, list)
        assert len(messages) == len(expected_messages)
        assert messages[1:] == expected_messages[1:]
        system_actual = messages[0]
        system_expected = expected_messages[0]
        assert isinstance(system_actual, dict)
        assert isinstance(system_expected, dict)
        assert system_actual.get("role") == system_expected.get("role")
        actual_content = system_actual.get("content")
        expected_content = system_expected.get("content")
        assert isinstance(actual_content, str)
        assert isinstance(expected_content, str)
        assert actual_content.startswith(expected_content.strip())
        assert "Debt rules:" in actual_content
        return {"decision": "merge", "reason": reason, "flags": dict(flags)}

    monkeypatch.setattr(send_ai_merge_packs, "decide_merge_or_different", _fake_decide)
    monkeypatch.setattr(
        send_ai_merge_packs,
        "_isoformat_timestamp",
        lambda dt=None: timestamp,
    )

    send_ai_merge_packs.main(["--sid", sid, "--runs-root", str(runs_root)])

    tags_a = json.loads((accounts_root / "1" / "tags.json").read_text(encoding="utf-8"))
    tags_b = json.loads((accounts_root / "2" / "tags.json").read_text(encoding="utf-8"))

    def _expected_flags(raw_flags: dict[str, bool | str]) -> dict[str, str]:
        normalized: dict[str, str] = {}
        for key, value in raw_flags.items():
            if isinstance(value, str):
                lowered = value.strip().lower()
                if lowered in {"true", "false", "unknown"}:
                    normalized[key] = lowered
                else:
                    normalized[key] = lowered
            else:
                normalized[key] = "true" if value else "false"
        return normalized

    expected_flags = _expected_flags(flags)

    ai_tag_a = next(tag for tag in tags_a if tag["kind"] == "ai_decision")
    assert ai_tag_a["decision"] == expected_decision
    assert ai_tag_a["with"] == 2
    assert ai_tag_a["reason"] == reason
    assert ai_tag_a["flags"] == expected_flags

    ai_tag_b = next(tag for tag in tags_b if tag["kind"] == "ai_decision")
    assert ai_tag_b["decision"] == expected_decision
    assert ai_tag_b["with"] == 1
    assert ai_tag_b["reason"] == reason
    assert ai_tag_b["flags"] == expected_flags

    def _pair_kinds(tags: list[dict[str, Any]]) -> set[str]:
        return {tag["kind"] for tag in tags if tag["kind"] in {"same_account_pair", "same_debt_pair"}}

    pair_kinds_a = _pair_kinds(tags_a)
    pair_kinds_b = _pair_kinds(tags_b)

    if expected_pair is None:
        assert pair_kinds_a == set()
        assert pair_kinds_b == set()
    else:
        assert pair_kinds_a == {expected_pair}
        assert pair_kinds_b == {expected_pair}
