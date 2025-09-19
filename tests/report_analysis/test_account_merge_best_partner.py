import json
import logging
from pathlib import Path

from backend.core.logic.report_analysis.account_merge import (
    choose_best_partner,
    persist_merge_tags,
    score_all_pairs_0_100,
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

    scores = score_all_pairs_0_100(sid, [0, 1, 2], runs_root=tmp_path)

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


def test_score_all_pairs_emits_structured_logs(tmp_path, caplog) -> None:
    sid = "SID-LOG"
    accounts_root = tmp_path / sid / "cases" / "accounts"

    bureaus_a = {
        "transunion": {
            "balance_owed": "1200",
            "account_number": "1234567890123456",
        }
    }
    bureaus_b = {
        "experian": {
            "balance_owed": 1200,
            "account_number": "1234567890123456",
        }
    }

    _write_account_payload(accounts_root, 0, bureaus_a)
    _write_account_payload(accounts_root, 1, bureaus_b)

    with caplog.at_level(
        logging.INFO, logger="backend.core.logic.report_analysis.account_merge"
    ):
        score_all_pairs_0_100(sid, [0, 1], runs_root=tmp_path)

    score_messages = [
        record.getMessage()
        for record in caplog.records
        if record.getMessage().startswith("MERGE_SCORE ")
    ]
    assert score_messages
    score_payload = json.loads(score_messages[0].split(" ", 1)[1])
    assert score_payload["sid"] == sid
    assert score_payload["i"] == 0
    assert score_payload["j"] == 1
    assert score_payload["parts"]["balance_owed"] > 0
    assert score_payload["acctnum_level"]
    assert "matched_pairs" in score_payload
    assert "account_number" in score_payload["matched_pairs"]

    trigger_messages = [
        record.getMessage()
        for record in caplog.records
        if record.getMessage().startswith("MERGE_TRIGGER ")
    ]
    assert trigger_messages
    trigger_payload = json.loads(trigger_messages[0].split(" ", 1)[1])
    assert {
        "sid",
        "i",
        "j",
        "kind",
        "details",
    }.issubset(trigger_payload)

    decision_messages = [
        record.getMessage()
        for record in caplog.records
        if record.getMessage().startswith("MERGE_DECISION ")
    ]
    assert decision_messages
    decision_payload = json.loads(decision_messages[0].split(" ", 1)[1])
    for key in ("sid", "i", "j", "decision", "total"):
        assert key in decision_payload
