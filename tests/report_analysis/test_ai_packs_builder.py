import json
import logging
from pathlib import Path

import pytest

from backend.core.logic.report_analysis.ai_packs import build_merge_ai_packs
from backend.pipeline.runs import RUNS_ROOT_ENV
from scripts.build_ai_merge_packs import main as build_packs_main


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_raw_lines(path: Path, lines: list[str]) -> None:
    payload = [{"text": text} for text in lines]
    _write_json(path, payload)


def _merge_pair_tag(partner: int) -> dict:
    return {
        "tag": "merge_pair",
        "kind": "merge_pair",
        "source": "merge_scorer",
        "with": partner,
        "decision": "ai",
        "total": 59,
        "mid": 20,
        "dates_all": False,
        "parts": {"balance_owed": 31, "account_number": 28},
        "aux": {
            "acctnum_level": "exact_or_known_match",
            "matched_fields": {
                "balance_owed": True,
                "last_payment": True,
                "account_number": True,
            },
        },
        "conflicts": ["credit_limit:conflict"],
        "strong": True,
    }


def _merge_best_tag(partner: int) -> dict:
    return {
        "tag": "merge_best",
        "kind": "merge_best",
        "source": "merge_scorer",
        "with": partner,
        "decision": "ai",
        "total": 59,
        "mid": 20,
        "parts": {"balance_owed": 31, "account_number": 28},
        "aux": {
            "acctnum_level": "exact_or_known_match",
            "matched_fields": {
                "balance_owed": True,
                "last_payment": True,
                "account_number": True,
            },
        },
        "conflicts": ["credit_limit:conflict"],
        "strong": True,
    }


def test_build_merge_ai_packs_curates_context_and_prompt(tmp_path: Path) -> None:
    sid = "sample-sid"
    runs_root = tmp_path
    accounts_root = runs_root / sid / "cases" / "accounts"

    account_a_dir = accounts_root / "11"
    account_b_dir = accounts_root / "16"

    _write_raw_lines(
        account_a_dir / "raw_lines.json",
        [
            "US BK CACS",
            "Transunion ® Experian ® Equifax ®",
            "Account # 409451****** -- 409451******",
            "Balance Owed: $12,091 -- $12,091",
            "Two-Year Payment History: 111100001111",
            "Creditor Remarks: Late due to pandemic",
        ],
    )

    _write_raw_lines(
        account_b_dir / "raw_lines.json",
        [
            "U S BANK",
            "Account # -- 409451******",
            "Balance Owed: -- $12,091 --",
            "Past Due Amount: --",
            "Last Payment: 13.9.2024",
            "Days Late - 7 Year History: 0000000",
        ],
    )

    _write_json(account_a_dir / "tags.json", [_merge_pair_tag(16), _merge_best_tag(16)])
    _write_json(account_b_dir / "tags.json", [_merge_best_tag(11)])

    packs = build_merge_ai_packs(sid, runs_root, max_lines_per_side=6)

    assert len(packs) == 1
    pack = packs[0]

    assert pack["pair"] == {"a": 11, "b": 16}
    context_a = pack["context"]["a"]
    assert context_a[0] == "US BK CACS"
    assert context_a[1] == "Lender normalized: US BANK"
    assert "Two-Year Payment History" not in " ".join(context_a)
    assert pack["ids"]["account_number_a"] == "409451******"
    assert pack["ids"]["account_number_b"] == "409451******"
    assert pack["ids"]["account_number_a_normalized"] == "409451"
    assert pack["ids"]["account_number_b_normalized"] == "409451"
    assert pack["ids"]["account_number_a_last4"] == "9451"
    assert pack["ids"]["account_number_b_last4"] == "9451"
    assert pack["highlights"]["total"] == 59
    assert pack["highlights"]["identity_score"] == 28
    assert pack["highlights"]["debt_score"] == 31
    assert pack["highlights"]["matched_fields"]["balance_owed"] is True
    assert pack["limits"]["max_lines_per_side"] == 6
    assert set(pack["tolerances_hint"].keys()) == {
        "amount_abs_usd",
        "amount_ratio",
        "last_payment_day_tol",
    }

    messages = pack["messages"]
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert "adjudicator" in messages[0]["content"]

    user_payload = json.loads(messages[1]["content"])
    assert user_payload["pair"] == {"a": 11, "b": 16}
    assert user_payload["numeric_match_summary"]["total"] == 59
    assert user_payload["numeric_match_summary"]["identity_score"] == 28
    assert user_payload["numeric_match_summary"]["debt_score"] == 31
    assert user_payload["ids"]["account_number_a_normalized"] == "409451"
    assert user_payload["ids"]["account_number_b_normalized"] == "409451"
    assert user_payload["ids"]["account_number_a_last4"] == "9451"
    assert user_payload["ids"]["account_number_b_last4"] == "9451"
    assert user_payload["output_contract"]["decision"] == [
        "merge",
        "same_debt",
        "same_debt_account_different",
        "same_account",
        "same_account_debt_different",
        "different",
    ]
    assert user_payload["output_contract"]["flags"] == {
        "account_match": ["true", "false", "unknown"],
        "debt_match": ["true", "false", "unknown"],
    }


def test_build_merge_ai_packs_only_merge_best_filter(tmp_path: Path) -> None:
    sid = "filter-sid"
    runs_root = tmp_path
    accounts_root = runs_root / sid / "cases" / "accounts"

    account_a_dir = accounts_root / "21"
    account_b_dir = accounts_root / "22"

    _write_raw_lines(account_a_dir / "raw_lines.json", ["Creditor A", "Account # 1111"])
    _write_raw_lines(account_b_dir / "raw_lines.json", ["Creditor B", "Account # 2222"])

    _write_json(account_a_dir / "tags.json", [_merge_pair_tag(22)])
    _write_json(account_b_dir / "tags.json", [])

    packs_only_best = build_merge_ai_packs(sid, runs_root, max_lines_per_side=3)
    assert packs_only_best == []

    packs_all = build_merge_ai_packs(
        sid,
        runs_root,
        only_merge_best=False,
        max_lines_per_side=3,
    )

    assert len(packs_all) == 1
    pack = packs_all[0]
    assert pack["pair"] == {"a": 21, "b": 22}
    assert len(pack["context"]["a"]) <= 3
    assert len(pack["context"]["b"]) <= 3


def test_build_merge_ai_packs_caps_context_to_env_limit(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("AI_PACK_MAX_LINES_PER_SIDE", "7")

    sid = "env-cap-sid"
    runs_root = tmp_path
    accounts_root = runs_root / sid / "cases" / "accounts"

    account_a_dir = accounts_root / "31"
    account_b_dir = accounts_root / "32"

    _write_raw_lines(
        account_a_dir / "raw_lines.json",
        [
            "US BK CACS",
            "Account # 9988",
            "Balance Owed: $500",
            "Last Payment: 2024-01-01",
            "Past Due Amount: $0",
            "High Balance: $800",
            "Creditor Type: Bank",
            "Account Type: Revolving",
            "Payment Amount: $50",
            "Credit Limit: $1000",
            "Creditor Remarks: Test remark",
            "Two-Year Payment History: 111111",
        ],
    )

    _write_raw_lines(
        account_b_dir / "raw_lines.json",
        [
            "U S BANK",
            "Account # 7766",
            "Balance Owed: $400",
            "Last Payment: 2023-12-15",
            "Past Due Amount: $25",
            "High Balance: $900",
            "Creditor Type: Bank",
            "Account Type: Revolving",
            "Payment Amount: $40",
            "Credit Limit: $950",
            "Creditor Remarks: Sample",
            "Days Late - 7 Year History: 0000",
        ],
    )

    _write_json(account_a_dir / "tags.json", [_merge_best_tag(32)])
    _write_json(account_b_dir / "tags.json", [_merge_best_tag(31)])

    packs = build_merge_ai_packs(
        sid,
        runs_root,
        only_merge_best=False,
        max_lines_per_side=12,
    )

    assert len(packs) == 1
    pack = packs[0]

    context_a = pack["context"]["a"]
    context_b = pack["context"]["b"]

    assert context_a[0] == "US BK CACS"
    assert context_b[0] == "U S BANK"
    assert len(context_a) <= 7
    assert len(context_b) <= 7
    assert pack["limits"]["max_lines_per_side"] == 7


def test_build_ai_merge_packs_cli_updates_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    sid = "cli-sid"
    runs_root = tmp_path / "runs"
    accounts_root = runs_root / sid / "cases" / "accounts"

    account_a_dir = accounts_root / "11"
    account_b_dir = accounts_root / "16"

    _write_raw_lines(
        account_a_dir / "raw_lines.json",
        [
            "US BK CACS",
            "Account # 409451******",
        ],
    )

    _write_raw_lines(
        account_b_dir / "raw_lines.json",
        [
            "US BANK",
            "Account # 409451******",
        ],
    )

    _write_json(account_a_dir / "tags.json", [_merge_pair_tag(16), _merge_best_tag(16)])
    _write_json(account_b_dir / "tags.json", [_merge_best_tag(11)])

    monkeypatch.setenv(RUNS_ROOT_ENV, str(runs_root))

    caplog.clear()
    with caplog.at_level(logging.INFO, logger="scripts.build_ai_merge_packs"):
        build_packs_main(
            [
                "--sid",
                sid,
                "--runs-root",
                str(runs_root),
                "--max-lines-per-side",
                "5",
            ]
        )
    log_messages = [record.getMessage() for record in caplog.records]

    out_dir = runs_root / sid / "ai_packs"
    index_path = out_dir / "index.json"
    manifest_path = runs_root / sid / "manifest.json"

    assert any(f"PACKS_DIR_USED sid={sid}" in message for message in log_messages)
    assert any(
        f"PACK_WRITTEN sid={sid} file=pair_011_016.jsonl" in message
        for message in log_messages
    )
    assert any("INDEX_WRITTEN sid=cli-sid" in message for message in log_messages)
    assert any("MANIFEST_AI_PACKS_UPDATED sid=cli-sid" in message for message in log_messages)

    index_payload = json.loads(index_path.read_text(encoding="utf-8"))
    assert index_payload["sid"] == sid
    assert index_payload["pairs_count"] == 1
    pairs = index_payload.get("packs", [])
    assert len(pairs) == 1
    pair_entry = pairs[0]
    assert pair_entry["a"] == 11
    assert pair_entry["b"] == 16
    assert pair_entry["pack_file"].startswith("pair_011_016")
    assert pair_entry["pack_file"].endswith(".jsonl")
    assert isinstance(pair_entry["lines_a"], int)
    assert isinstance(pair_entry["lines_b"], int)
    assert isinstance(pair_entry["score_total"], int)

    logs_path = out_dir / "logs.txt"
    assert not logs_path.exists()
    assert manifest_path.exists()

    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    ai_section = manifest_payload["ai"]
    packs_info = ai_section["packs"]
    status_info = ai_section["status"]

    assert packs_info["dir"] == str(out_dir.resolve())
    assert packs_info["index"] == str(index_path.resolve())
    assert packs_info["pairs"] == 1
    assert status_info["built"] is True
