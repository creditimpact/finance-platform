"""Regression tests for cross-bureau account-number heuristics."""
from __future__ import annotations

import json
import logging
from pathlib import Path

from backend.core.logic.report_analysis import account_merge


def test_account_number_cross_bureau_prefers_transunion_visible_match() -> None:
    """Pairs use the strongest bureau digits and record the winning pair."""

    cfg = account_merge.get_merge_cfg()
    account_a = {
        "transunion": {"account_number_display": "349992123456"},
        "experian": {"account_number_display": "***123456"},
        "equifax": {"account_number_display": "-34999***********"},
    }
    account_b = {
        "transunion": {"account_number_display": "AA349992123456999"},
        "experian": {"account_number_display": "************"},
        "equifax": {"account_number_display": "************"},
    }

    result = account_merge.score_pair_0_100(account_a, account_b, cfg)

    acct_aux = result["aux"]["account_number"]
    assert acct_aux["acctnum_level"] == "exact_or_known_match"
    assert acct_aux["best_pair"] == ("transunion", "transunion")

    aux_payload = account_merge._build_aux_payload(result["aux"])
    assert aux_payload["by_field_pairs"]["account_number"] == [
        "transunion",
        "transunion",
    ]


def _write_account_payload(root: Path, idx: int, bureaus: dict[str, object]) -> None:
    account_dir = root / str(idx)
    account_dir.mkdir(parents=True, exist_ok=True)
    (account_dir / "bureaus.json").write_text(
        json.dumps(bureaus, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (account_dir / "summary.json").write_text(
        json.dumps({"account_index": idx}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (account_dir / "raw_lines.json").write_text("[]\n", encoding="utf-8")


def test_candidate_without_gates_is_skipped(tmp_path, caplog) -> None:
    sid = "SID-SOFT-ACCT"
    accounts_root = tmp_path / sid / "cases" / "accounts"

    bureaus_a = {"transunion": {"account_number_display": "1111167890"}}
    bureaus_b = {"transunion": {"account_number_display": "2222267890"}}

    _write_account_payload(accounts_root, 0, bureaus_a)
    _write_account_payload(accounts_root, 1, bureaus_b)

    with caplog.at_level(
        logging.INFO, logger="backend.core.logic.report_analysis.account_merge"
    ):
        scores = account_merge.score_all_pairs_0_100(sid, [0, 1], runs_root=tmp_path)

    assert scores[0] == {}

    consider_logs = [
        record.getMessage()
        for record in caplog.records
        if record.getMessage().startswith("CANDIDATE_CONSIDERED ")
    ]
    assert any("hard=False" in message for message in consider_logs)
    assert any("total=False" in message for message in consider_logs)
