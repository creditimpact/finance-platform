import json
from pathlib import Path

import pytest

from backend.core.logic.report_analysis import account_merge


def _write_account_payload(base: Path, idx: int, bureaus: dict) -> None:
    account_dir = base / str(idx)
    account_dir.mkdir(parents=True, exist_ok=True)
    (account_dir / "bureaus.json").write_text(
        json.dumps(bureaus, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (account_dir / "summary.json").write_text(
        json.dumps({"account_index": idx}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (account_dir / "raw_lines.json").write_text("[]\n", encoding="utf-8")


def _all_pairs(scores: dict[int, dict[int, dict]]) -> set[tuple[int, int]]:
    pairs: set[tuple[int, int]] = set()
    for left, partner_map in scores.items():
        for right in partner_map:
            if left == right:
                continue
            pair = (left, right) if left < right else (right, left)
            pairs.add(pair)
    return pairs


def test_hard_pairs_bypass_per_account_caps(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("MAX_CANDIDATES_PER_ACCOUNT", "1")
    monkeypatch.setenv("MERGE_CANDIDATE_LIMIT", "1")

    sid = "SID-HARD"
    accounts_root = tmp_path / sid / "cases" / "accounts"

    bureaus_28 = {
        "transunion": {
            "account_number_display": "349992*****",
            "balance_owed": "0",
        }
    }
    bureaus_29 = {
        "experian": {
            "account_number_display": "3499921234567",
            "balance_owed": "0",
        }
    }
    bureaus_39 = {
        "equifax": {
            "account_number_display": "3499921234567",
            "balance_owed": "0",
        }
    }

    _write_account_payload(accounts_root, 28, bureaus_28)
    _write_account_payload(accounts_root, 29, bureaus_29)
    _write_account_payload(accounts_root, 39, bureaus_39)

    scores = account_merge.score_all_pairs_0_100(sid, [28, 29, 39], runs_root=tmp_path)

    built_pairs = _all_pairs(scores)
    assert built_pairs == {(28, 29), (28, 39), (29, 39)}

    cfg = account_merge.get_merge_cfg()
    acct_weight = pytest.approx(
        getattr(cfg, "MERGE_WEIGHTS", {}).get("account_number", 0.0)
    )

    for left, right in built_pairs:
        result = scores[left][right]
        acct_aux = result["aux"]["account_number"]
        assert acct_aux["acctnum_level"] == "exact_or_known_match"
        assert result["field_contributions"]["account_number"] == acct_weight


def test_account_number_points_open_score_gate(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("AI_THRESHOLD", "27")

    sid = "SID-THRESHOLD"
    accounts_root = tmp_path / sid / "cases" / "accounts"

    bureaus_a = {
        "transunion": {
            "account_number_display": "123456789",
            "balance_owed": "0",
        }
    }
    bureaus_b = {
        "experian": {
            "account_number_display": "123456789",
            "balance_owed": "0",
        }
    }

    _write_account_payload(accounts_root, 0, bureaus_a)
    _write_account_payload(accounts_root, 1, bureaus_b)

    scores = account_merge.score_all_pairs_0_100(sid, [0, 1], runs_root=tmp_path)

    result = scores[0][1]
    acct_aux = result["aux"]["account_number"]

    cfg = account_merge.get_merge_cfg()
    acct_weight_value = getattr(cfg, "MERGE_WEIGHTS", {}).get("account_number", 0.0)
    balance_weight_value = getattr(cfg, "MERGE_WEIGHTS", {}).get("balance_owed", 0.0)

    expected_total = acct_weight_value + balance_weight_value

    assert result["total"] == pytest.approx(expected_total)
    assert acct_aux["acctnum_level"] == "exact_or_known_match"
    assert result["triggers"] == []
