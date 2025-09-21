import json
from pathlib import Path

import pytest

from scripts.score_bureau_pairs import (
    build_merge_tags,
    build_pair_rows,
    compute_scores_for_sid,
    score_accounts,
)


def _write_bureaus(base: Path, payload: dict) -> None:
    base.mkdir(parents=True, exist_ok=True)
    (base / "bureaus.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (base / "raw_lines.json").write_text("[]\n", encoding="utf-8")


@pytest.fixture
def runs_root(tmp_path: Path) -> Path:
    return tmp_path / "runs"


def test_score_bureau_pairs_cli_helpers(runs_root: Path) -> None:
    sid = "SID001"
    accounts_dir = runs_root / sid / "cases" / "accounts"

    common_tu = {
        "balance_owed": "1500",
        "account_number": "1111222233334444",
        "last_payment": "2022-01-10",
        "past_due_amount": "100",
        "high_balance": "2000",
        "creditor_type": "credit card",
        "account_type": "revolving",
        "payment_amount": "50",
        "credit_limit": "2000",
        "last_verified": "2022-01-20",
        "date_of_last_activity": "2022-01-05",
        "date_reported": "2022-01-22",
        "date_opened": "2020-01-01",
        "closed_date": "2023-01-01",
    }

    bureaus_a = {
        "transunion": dict(common_tu),
        "experian": dict(common_tu),
        "equifax": dict(common_tu),
    }

    bureaus_b = {
        "transunion": dict(common_tu),
        "experian": dict(common_tu),
        "equifax": dict(common_tu),
    }

    _write_bureaus(accounts_dir / "1", bureaus_a)
    _write_bureaus(accounts_dir / "2", bureaus_b)

    indices, scores = compute_scores_for_sid(sid, runs_root=runs_root)

    assert indices == [1, 2]
    assert scores[1][2]["decision"] == "auto"
    assert scores[1][2]["total"] >= 70

    rows = build_pair_rows(scores)
    assert len(rows) == 1
    row = rows[0]
    assert row["i"] == 1 and row["j"] == 2
    assert row["decision"] == "auto"
    assert row["strong_flag"] is True
    assert row["acctnum_level"] == "exact"
    assert "balance_owed" in row["parts"]
    assert row["parts"]["balance_owed"] == 31
    assert row["matched_pairs_map"]["balance_owed"] == ["transunion", "transunion"]

    merge_tags = build_merge_tags(scores)
    assert merge_tags[1]["decision"] == "auto"
    assert merge_tags[1]["score_total"] >= 70
    assert merge_tags[1]["aux"]["acctnum_level"] == "exact"
    assert merge_tags[2]["decision"] == "auto"


def test_score_accounts_writes_merge_tags(runs_root: Path) -> None:
    sid = "SID002"
    accounts_dir = runs_root / sid / "cases" / "accounts"

    bureaus_payload = {
        "transunion": {"balance_owed": "1000", "account_number": "9999"},
        "experian": {"balance_owed": "1000", "account_number": "9999"},
    }

    _write_bureaus(accounts_dir / "5", bureaus_payload)
    _write_bureaus(accounts_dir / "9", bureaus_payload)

    result = score_accounts(sid, runs_root=runs_root, write_tags=True)

    assert result.indices == [5, 9]
    assert result.merge_tags
    assert result.best_by_idx[5]["partner_index"] == 9

    tags_path = accounts_dir / "5" / "tags.json"
    assert tags_path.exists()
    tag_data = json.loads(tags_path.read_text(encoding="utf-8"))
    merge_pairs = [entry for entry in tag_data if entry.get("kind") == "merge_pair"]
    assert merge_pairs
    assert merge_pairs[0]["with"] == 9
    assert merge_pairs[0]["decision"] in {"ai", "auto"}

