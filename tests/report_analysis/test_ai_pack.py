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

    pack_a_path = account_a_dir / "ai" / "pack_pair_11_16.json"
    pack_b_path = account_b_dir / "ai" / "pack_pair_16_11.json"

    assert pack_a_path.exists()
    assert pack_b_path.exists()

    saved_a = json.loads(pack_a_path.read_text(encoding="utf-8"))
    saved_b = json.loads(pack_b_path.read_text(encoding="utf-8"))

    assert saved_a == pack
    assert saved_b["pair"] == {"a": 16, "b": 11}
    assert saved_b["context"]["a"][0] == "U S BANK"
    assert saved_b["context"]["b"][0] == "US BK CACS"
