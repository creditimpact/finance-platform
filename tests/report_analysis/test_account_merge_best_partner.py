import json
import logging
from pathlib import Path

from backend.core.logic.report_analysis import account_merge
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
    (account_dir / "raw_lines.json").write_text("[]\n", encoding="utf-8")


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
    assert "merge_tag" not in summary_data

    tags_path = accounts_root / "0" / "tags.json"
    tags_payload = json.loads(tags_path.read_text(encoding="utf-8"))
    pair_tags = [tag for tag in tags_payload if tag.get("kind") == "merge_pair"]
    by_partner = {tag.get("with"): tag for tag in pair_tags}
    assert all(tag.get("decision") in {"ai", "auto"} for tag in pair_tags)

    partner_decision = scores[0][2]["decision"]
    if partner_decision in {"ai", "auto"}:
        assert by_partner[2]["decision"] == partner_decision
    else:
        assert 2 not in by_partner

    best_tags = [tag for tag in tags_payload if tag.get("kind") == "merge_best"]
    if partner_decision in {"ai", "auto"}:
        assert best_tags and best_tags[0]["with"] == 2
        assert best_tags[0]["decision"] == partner_decision
    else:
        assert not best_tags


def test_persist_merge_tags_updates_tags_when_legacy_disabled(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("MERGE_V2_ONLY", "0")

    sid = "SID-LEGACY"
    accounts_root = tmp_path / sid / "cases" / "accounts"

    bureaus_a = {"transunion": {"balance_owed": "100"}}
    bureaus_b = {"experian": {"balance_owed": "100"}}

    _write_account_payload(accounts_root, 0, bureaus_a)
    _write_account_payload(accounts_root, 1, bureaus_b)

    scores = score_all_pairs_0_100(sid, [0, 1], runs_root=tmp_path)
    best = choose_best_partner(scores)

    merge_tags = persist_merge_tags(sid, scores, best, runs_root=tmp_path)
    tag_a = merge_tags[0]

    summary_path = accounts_root / "0" / "summary.json"
    summary_data = json.loads(summary_path.read_text(encoding="utf-8"))
    assert "merge_tag" not in summary_data

    tags_path = accounts_root / "0" / "tags.json"
    tags_payload = json.loads(tags_path.read_text(encoding="utf-8"))
    merge_pairs = [tag for tag in tags_payload if tag.get("kind") == "merge_pair"]
    assert merge_pairs
    assert merge_pairs[0]["decision"] in {"ai", "auto"}


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

    v2_score_messages = [
        record.getMessage()
        for record in caplog.records
        if record.getMessage().startswith("MERGE_V2_SCORE ")
    ]
    assert v2_score_messages
    score_payload = json.loads(v2_score_messages[0].split(" ", 1)[1])
    assert score_payload["sid"] == sid
    assert score_payload["i"] == 0
    assert score_payload["j"] == 1
    assert score_payload["parts"]["balance_owed"] > 0
    assert score_payload["acctnum_level"]
    assert "matched_pairs" in score_payload
    assert "account_number" in score_payload["matched_pairs"]

    v2_trigger_messages = [
        record.getMessage()
        for record in caplog.records
        if record.getMessage().startswith("MERGE_V2_TRIGGER ")
    ]
    assert v2_trigger_messages
    trigger_payload = json.loads(v2_trigger_messages[0].split(" ", 1)[1])
    assert {
        "sid",
        "i",
        "j",
        "kind",
        "details",
    }.issubset(trigger_payload)

    v2_decision_messages = [
        record.getMessage()
        for record in caplog.records
        if record.getMessage().startswith("MERGE_V2_DECISION ")
    ]
    assert v2_decision_messages
    decision_payload = json.loads(v2_decision_messages[0].split(" ", 1)[1])
    for key in ("sid", "i", "j", "decision", "total"):
        assert key in decision_payload


def test_score_all_pairs_debug_pair_logs(tmp_path, caplog) -> None:
    sid = "SID-DEBUG"
    accounts_root = tmp_path / sid / "cases" / "accounts"

    for idx in range(3):
        bureaus = {"transunion": {"balance_owed": str(100 + idx)}}
        _write_account_payload(accounts_root, idx, bureaus)

    with caplog.at_level(
        logging.DEBUG, logger="backend.core.logic.report_analysis.account_merge"
    ):
        score_all_pairs_0_100(sid, [0, 1, 2], runs_root=tmp_path)

    step_payloads = [
        json.loads(record.getMessage().split(" ", 1)[1])
        for record in caplog.records
        if record.getMessage().startswith("MERGE_PAIR_STEP ")
    ]
    summary_payloads = [
        json.loads(record.getMessage().split(" ", 1)[1])
        for record in caplog.records
        if record.getMessage().startswith("MERGE_PAIR_SUMMARY ")
    ]

    assert step_payloads
    assert summary_payloads

    summary_payload = summary_payloads[-1]
    expected_pairs = {(0, 1), (0, 2), (1, 2)}
    logged_pairs = {(payload["i"], payload["j"]) for payload in step_payloads}

    assert logged_pairs == expected_pairs
    assert summary_payload["expected_pairs"] == len(expected_pairs)
    assert summary_payload["pairs_scored"] == len(step_payloads) == len(expected_pairs)


def test_candidate_loop_logs_and_soft_gate(tmp_path, caplog) -> None:
    sid = "SID-SOFT"
    accounts_root = tmp_path / sid / "cases" / "accounts"

    bureaus_a = {
        "transunion": {
            "account_number_display": "0000012345",
            "balance_owed": "200",
        }
    }
    bureaus_b = {
        "equifax": {
            "account_number_display": "9912345",
            "balance_owed": "350",
        }
    }

    _write_account_payload(accounts_root, 0, bureaus_a)
    _write_account_payload(accounts_root, 1, bureaus_b)

    with caplog.at_level(
        logging.INFO, logger="backend.core.logic.report_analysis.account_merge"
    ):
        scores = score_all_pairs_0_100(sid, [0, 1], runs_root=tmp_path)

    assert 1 not in scores.get(0, {})
    assert scores[0] == {}
    assert scores[1] == {}

    start_messages = [
        record.getMessage()
        for record in caplog.records
        if record.getMessage().startswith("CANDIDATE_LOOP_START ")
    ]
    end_messages = [
        record.getMessage()
        for record in caplog.records
        if record.getMessage().startswith("CANDIDATE_LOOP_END ")
    ]

    assert start_messages == ["CANDIDATE_LOOP_START sid=SID-SOFT total_accounts=2"]
    assert end_messages == ["CANDIDATE_LOOP_END sid=SID-SOFT built_pairs=0"]


def test_candidate_limit_env_is_ignored(tmp_path, monkeypatch, caplog) -> None:
    monkeypatch.setenv("MERGE_CANDIDATE_LIMIT", "1")

    sid = "SID-LIMIT"
    accounts_root = tmp_path / sid / "cases" / "accounts"

    bureaus_zero = {
        "transunion": {
            "account_number": "1234567890123456",
        }
    }
    bureaus_one = {
        "experian": {
            "account_number_display": "1234567890123456",
        }
    }
    bureaus_two = {
        "equifax": {
            "account_number_display": "1234567890123456",
        }
    }

    _write_account_payload(accounts_root, 0, bureaus_zero)
    _write_account_payload(accounts_root, 1, bureaus_one)
    _write_account_payload(accounts_root, 2, bureaus_two)

    with caplog.at_level(
        logging.INFO, logger="backend.core.logic.report_analysis.account_merge"
    ):
        scores = score_all_pairs_0_100(sid, [0, 1, 2], runs_root=tmp_path)

    assert 1 in scores[0]
    assert 2 in scores[0]
    assert 0 in scores[1]
    assert 2 in scores[1]
    assert 0 in scores[2]
    assert 1 in scores[2]

    loop_end_messages = [
        record.getMessage()
        for record in caplog.records
        if record.getMessage().startswith("CANDIDATE_LOOP_END ")
    ]
    assert loop_end_messages == ["CANDIDATE_LOOP_END sid=SID-LIMIT built_pairs=3"]


def test_score_pairs_respect_normalized_hard_gate(tmp_path, monkeypatch) -> None:
    sid = "SID-HARD-GATE"
    accounts_root = tmp_path / sid / "cases" / "accounts"

    for idx in (14, 29, 30):
        _write_account_payload(
            accounts_root,
            idx,
            {"transunion": {"account_number": f"{idx:03d}"}},
        )

    def fake_load_bureaus(_sid: str, idx: int, *, runs_root: Path) -> dict:
        return {"meta": {"idx": idx}}

    def fake_score_pair(left_bureaus: dict, right_bureaus: dict, _cfg: object) -> dict:
        left_idx = left_bureaus["meta"]["idx"]
        right_idx = right_bureaus["meta"]["idx"]
        return {
            "total": 0,
            "parts": {"account_number": account_merge.POINTS_ACCTNUM_VISIBLE},
            "aux": {
                "account_number": {
                    "acctnum_level": "exact_or_known_match",
                    "raw_values": {
                        "a": f"{left_idx:03d}",
                        "b": f"{right_idx:03d}",
                    },
                }
            },
            "decision": "ai",
            "triggers": [],
        }

    monkeypatch.setattr(account_merge, "load_bureaus", fake_load_bureaus)
    monkeypatch.setattr(account_merge, "score_pair_0_100", fake_score_pair)

    scores = account_merge.score_all_pairs_0_100(
        sid,
        [14, 29, 30],
        runs_root=tmp_path,
    )

    assert sorted(scores[14]) == [29, 30]
    assert sorted(scores[29]) == [14, 30]
    assert sorted(scores[30]) == [14, 29]

    for left, right in ((14, 29), (14, 30), (29, 30)):
        result = scores[left][right]
        assert result["total"] == 0
        assert (
            result.get("aux", {})
            .get("account_number", {})
            .get("acctnum_level")
            == "exact_or_known_match"
        )
