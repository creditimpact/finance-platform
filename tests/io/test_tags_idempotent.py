from __future__ import annotations

import json
from pathlib import Path

from backend.core.logic.report_analysis.problem_case_builder import build_problem_cases


SID = "IDEMP-301"


def _write_stagea_accounts(base: Path, accounts: list[dict]) -> None:
    table_dir = base / "traces" / "blocks" / SID / "accounts_table"
    table_dir.mkdir(parents=True, exist_ok=True)
    payload = {"accounts": accounts}
    (table_dir / "accounts_from_full.json").write_text(
        json.dumps(payload, ensure_ascii=False),
        encoding="utf-8",
    )


def _read_tags_snapshot(run_root: Path, indices: list[int]) -> tuple[dict[int, str], dict[int, list[dict]]]:
    tags_dir = run_root / "cases" / "accounts"
    text_map: dict[int, str] = {}
    data_map: dict[int, list[dict]] = {}

    for idx in indices:
        path = tags_dir / str(idx) / "tags.json"
        content = path.read_text(encoding="utf-8")
        text_map[idx] = content
        data_map[idx] = json.loads(content)

    return text_map, data_map


def _assert_tag_integrity(tags: dict[int, list[dict]], *, issue_by_index: dict[int, str]) -> None:
    for idx, entries in tags.items():
        serialized = [json.dumps(entry, sort_keys=True) for entry in entries]
        assert len(serialized) == len(set(serialized)), f"duplicate tags for account {idx}"

        issues = [entry for entry in entries if entry.get("kind") == "issue"]
        assert len(issues) == 1, f"missing issue tag for account {idx}"
        assert issues[0].get("type") == issue_by_index[idx]

        pairs = [entry for entry in entries if entry.get("kind") == "merge_pair"]
        bests = [entry for entry in entries if entry.get("kind") == "merge_best"]

        if idx in (0, 1):
            assert len(pairs) == 1 and len(bests) == 1
            partner = 1 if idx == 0 else 0
            assert pairs[0].get("with") == partner
            assert pairs[0].get("decision") in {"ai", "auto"}
            assert bests[0].get("with") == partner
            assert bests[0].get("decision") in {"ai", "auto"}
        else:
            assert pairs == []
            assert bests == []


def _build_accounts_payload() -> list[dict]:
    base_account = {
        "lines": [],
        "two_year_payment_history": {},
        "seven_year_history": {},
        "triad": {"order": ["transunion", "experian", "equifax"]},
    }

    account_a = {
        **base_account,
        "account_index": 0,
        "account_id": "0",
        "triad_fields": {
            "transunion": {
                "balance_owed": "100",
                "last_payment": "2024-01-01",
                "past_due_amount": "50",
                "high_balance": "500",
                "account_type": "Credit Card",
                "date_of_last_activity": "2024-01-02",
                "date_opened": "2020-01-01",
            }
        },
    }

    account_b = {
        **base_account,
        "account_index": 1,
        "account_id": "1",
        "triad_fields": {
            "experian": {
                "balance_owed": "100",
                "last_payment": "2024-01-01",
                "past_due_amount": "50",
                "high_balance": "500",
                "account_type": "Credit Card",
                "date_of_last_activity": "2024-01-02",
                "date_opened": "2020-01-01",
            }
        },
    }

    account_c = {
        **base_account,
        "account_index": 2,
        "account_id": "2",
        "triad_fields": {
            "equifax": {
                "balance_owed": "9000",
                "last_payment": "2021-06-01",
                "past_due_amount": "0",
                "high_balance": "10000",
                "account_type": "Mortgage",
                "date_of_last_activity": "2021-05-02",
                "date_opened": "2015-01-01",
            }
        },
    }

    return [account_a, account_b, account_c]


def _build_candidates() -> list[dict]:
    return [
        {"account_index": 0, "account_id": "0", "primary_issue": "collection"},
        {"account_index": 1, "account_id": "1", "primary_issue": "collection"},
        {"account_index": 2, "account_id": "2", "primary_issue": "late_payment"},
    ]


def test_tags_are_idempotent_across_pipeline_runs(tmp_path: Path) -> None:
    accounts = _build_accounts_payload()
    _write_stagea_accounts(tmp_path, accounts)
    candidates = _build_candidates()

    def run_pipeline() -> None:
        build_problem_cases(SID, candidates=candidates, root=tmp_path)

    run_pipeline()
    run_dir = tmp_path / "runs" / SID
    indices = [account["account_index"] for account in accounts]
    text_first, data_first = _read_tags_snapshot(run_dir, indices)

    _assert_tag_integrity(data_first, issue_by_index={0: "collection", 1: "collection", 2: "late_payment"})

    run_pipeline()
    text_second, data_second = _read_tags_snapshot(run_dir, indices)

    assert text_second == text_first
    assert data_second == data_first

    _assert_tag_integrity(data_second, issue_by_index={0: "collection", 1: "collection", 2: "late_payment"})
