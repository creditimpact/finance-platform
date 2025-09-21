from __future__ import annotations

import json
from pathlib import Path

from backend.core.logic.report_analysis.ai_pack import build_ai_pack_for_pair


def _write_raw_lines(path: Path, lines: list[str]) -> None:
    payload = [{"text": text} for text in lines]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_build_ai_pack_for_pair_creates_packs(tmp_path, monkeypatch):
    monkeypatch.setenv("AI_PACK_MAX_LINES_PER_SIDE", "5")

    sid = "sample-sid"
    runs_root = tmp_path
    accounts_root = runs_root / sid / "cases" / "accounts"

    account_a_dir = accounts_root / "11"
    account_b_dir = accounts_root / "16"

    raw_lines_a = [
        "US BK CACS",
        "Transunion ® Experian ® Equifax ®",
        "Account # 409451****** -- 409451******",
        "Balance Owed: $12,091 -- $12,091",
        "--",
        "Last Payment: 13.9.2024 -- 1.11.2024",
    ]
    raw_lines_b = [
        "U S BANK",
        "Transunion ® Experian ® Equifax ®",
        "Account # -- 409451******",
        "Balance Owed: -- $12,091 --",
        "Past Due Amount: --",
        "Last Payment: -- 13.9.2024 --",
    ]

    _write_raw_lines(account_a_dir / "raw_lines.json", raw_lines_a)
    _write_raw_lines(account_b_dir / "raw_lines.json", raw_lines_b)

    highlights = {
        "total": 59,
        "triggers": ["strong", "mid", "total"],
        "parts": {"balance_owed": 31, "account_number": 0},
        "matched_fields": {"balance_owed": True},
        "conflicts": ["amount_conflict:high_balance"],
        "acctnum_level": "none",
    }

    pack = build_ai_pack_for_pair(sid, runs_root, 11, 16, highlights)

    expected_context_a = [
        "US BK CACS",
        "Account # 409451****** -- 409451******",
        "Balance Owed: $12,091 -- $12,091",
        "Last Payment: 13.9.2024 -- 1.11.2024",
    ]
    assert pack["context"]["a"] == expected_context_a
    assert pack["context"]["b"][0] == "U S BANK"
    assert all("Transunion" not in line for line in pack["context"]["a"])
    assert "--" not in {line.strip() for line in pack["context"]["a"]}
    assert len(pack["context"]["a"]) <= 5
    assert len(pack["context"]["b"]) <= 5
    assert pack["ids"]["account_number_a"] == "409451******"
    assert pack["ids"]["account_number_b"] == "409451******"
    assert pack["limits"]["max_lines_per_side"] == 5

    assert not (account_a_dir / "ai").exists()
    assert not (account_b_dir / "ai").exists()


def test_build_ai_pack_for_pair_is_consistent(tmp_path, monkeypatch):
    monkeypatch.setenv("AI_PACK_MAX_LINES_PER_SIDE", "3")

    sid = "sample-symmetric"
    runs_root = tmp_path
    accounts_root = runs_root / sid / "cases" / "accounts"

    account_a_dir = accounts_root / "201"
    account_b_dir = accounts_root / "305"

    _write_raw_lines(
        account_a_dir / "raw_lines.json",
        ["Creditor A", "Account # 1111", "Balance Owed: $100"],
    )
    _write_raw_lines(
        account_b_dir / "raw_lines.json",
        ["Creditor B", "Account # 2222", "Balance Owed: $200"],
    )

    highlights = {"total": 42, "triggers": ["strong:balance"]}

    first_pack = build_ai_pack_for_pair(sid, runs_root, 201, 305, highlights)

    assert first_pack["pair"] == {"a": 201, "b": 305}
    assert first_pack["context"]["a"][0] == "Creditor A"
    assert first_pack["context"]["b"][0] == "Creditor B"

    second_pack = build_ai_pack_for_pair(sid, runs_root, 305, 201, highlights)

    assert second_pack["pair"] == {"a": 305, "b": 201}
    assert second_pack["context"]["a"][0] == "Creditor B"
    assert second_pack["context"]["b"][0] == "Creditor A"

    assert not (account_a_dir / "ai").exists()
    assert not (account_b_dir / "ai").exists()
