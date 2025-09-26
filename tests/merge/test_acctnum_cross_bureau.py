"""Regression tests for cross-bureau account-number heuristics."""
from __future__ import annotations

import json
import logging
from pathlib import Path

from backend.core.logic.report_analysis import account_merge


def test_account_number_cross_bureau_prefers_transunion_last6_bin() -> None:
    """Pairs use the strongest bureau digits and record the winning pair."""

    cfg = account_merge.get_merge_cfg()
    account_a = {
        "transunion": {"account_number_display": "349992999999123456"},
        "experian": {"account_number_display": "349992999999123456"},
        "equifax": {"account_number_display": "-34999***********"},
    }
    account_b = {
        "transunion": {"account_number_display": "349992111111123456"},
        "experian": {"account_number_display": "349992111111123456"},
        "equifax": {"account_number_display": "************"},
    }

    result = account_merge.score_pair_0_100(account_a, account_b, cfg)

    acct_aux = result["aux"]["account_number"]
    assert acct_aux["acctnum_level"] == "last6_bin"
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


def test_soft_last5_candidate_is_admitted(tmp_path, caplog) -> None:
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

    assert 1 in scores[0]

    result = scores[0][1]
    assert result["parts"]["account_number"] == 0
    assert result["aux"]["account_number"]["acctnum_level"] == "none"

    consider_logs = [
        record.getMessage()
        for record in caplog.records
        if record.getMessage().startswith("CANDIDATE_CONSIDERED ")
    ]
    assert any("soft_last5=True" in message for message in consider_logs)
    assert any("hard=False" in message for message in consider_logs)
