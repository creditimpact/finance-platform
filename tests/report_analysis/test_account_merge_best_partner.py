import json
from pathlib import Path

from backend.core.logic.report_analysis.account_merge import (
    choose_best_partner,
    persist_merge_tags,
    score_all_pairs,
)


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


def test_best_partner_prefers_strong_match(tmp_path) -> None:
    sid = "SID-123"
    accounts_root = tmp_path / sid / "cases" / "accounts"

    bureaus_a = {
        "transunion": {
            "balance_owed": "100",
            "last_payment": "2024-01-01",
            "past_due_amount": "50",
            "high_balance": "500",
            "account_type": "Credit Card",
            "date_of_last_activity": "2024-01-02",
            "date_opened": "2020-01-01",
        }
    }
    bureaus_b = {
        "experian": {
            "balance_owed": "150",
            "last_payment": "2024-01-05",
            "past_due_amount": "50",
            "high_balance": "500",
            "account_type": "Credit Card",
            "date_of_last_activity": "2024-01-02",
            "date_opened": "2020-01-01",
        }
    }
    bureaus_c = {
        "equifax": {
            "balance_owed": "100",
        }
    }

    _write_account_payload(accounts_root, 0, bureaus_a)
    _write_account_payload(accounts_root, 1, bureaus_b)
    _write_account_payload(accounts_root, 2, bureaus_c)

    scores = score_all_pairs(sid, [0, 1, 2], runs_root=tmp_path)

    assert scores[0][1]["total"] > scores[0][2]["total"]
    assert "strong:balance_owed" in scores[0][2]["triggers"]
    assert "strong:balance_owed" not in scores[0][1]["triggers"]

    best = choose_best_partner(scores)

    assert best[0]["partner_index"] == 2
    assert best[0]["tiebreaker"] == "strong"

    merge_tags = persist_merge_tags(sid, scores, best, runs_root=tmp_path)

    tag_a = merge_tags[0]
    assert tag_a["decision"] == scores[0][2]["decision"]
    assert tag_a["score_total"] == scores[0][2]["total"]
    assert tag_a["score_to"][0]["account_index"] == 2
    assert tag_a["score_to"][1]["account_index"] == 1
    assert tag_a["tiebreaker"] == "strong"
    assert tag_a["aux"]["by_field_pairs"]["balance_owed"] == [
        "transunion",
        "equifax",
    ]

    summary_path = accounts_root / "0" / "summary.json"
    summary_data = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary_data["merge_tag"] == tag_a
