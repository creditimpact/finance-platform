import json
from pathlib import Path

from backend.core.logic.report_analysis.ai_packs import build_merge_ai_packs


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
            "acctnum_level": "last4",
            "matched_fields": {"balance_owed": True, "last_payment": True},
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
            "acctnum_level": "last4",
            "matched_fields": {"balance_owed": True, "last_payment": True},
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
    assert pack["highlights"]["total"] == 59
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
    assert user_payload["output_contract"]["decision"] == [
        "merge",
        "same_debt",
        "different",
    ]


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
