import json
from pathlib import Path

from backend.core.logic.report_analysis import ai_sender

from scripts.run_ai_merge_flow import (
    _load_ai_outcomes,
    prepare_summary_rows,
)
from scripts.score_bureau_pairs import score_accounts


def _write_account_payload(base: Path, idx: int, bureaus: dict) -> None:
    account_dir = base / str(idx)
    account_dir.mkdir(parents=True, exist_ok=True)
    (account_dir / "bureaus.json").write_text(
        json.dumps(bureaus, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (account_dir / "raw_lines.json").write_text("[]\n", encoding="utf-8")


def test_prepare_summary_rows_with_ai_decision(tmp_path: Path) -> None:
    sid = "SID-RUN-1"
    runs_root = tmp_path / "runs"
    accounts_dir = runs_root / sid / "cases" / "accounts"

    bureaus_payload = {
        "transunion": {
            "balance_owed": "1500",
            "account_number": "123456",
            "date_opened": "2020-01-01",
        }
    }

    _write_account_payload(accounts_dir, 1, bureaus_payload)
    _write_account_payload(accounts_dir, 2, bureaus_payload)

    computation = score_accounts(sid, runs_root=runs_root, write_tags=True)

    ai_sender.write_decision_tags(
        runs_root,
        sid,
        1,
        2,
        "same_debt",
        "matching balances",
        "2024-01-01T00:00:00Z",
        {"account_match": "unknown", "debt_match": True},
    )

    outcomes = _load_ai_outcomes(sid, runs_root)
    rows = prepare_summary_rows(computation, outcomes)

    assert rows
    row = rows[0]
    assert row["idx"] in {1, 2}
    assert row["best_with"] in {1, 2}
    assert row["pre_decision"] in {"ai", "auto"}
    assert row["ai_decision"] == "same_debt"
    assert row["ai_reason"] == "matching balances"


def test_prepare_summary_rows_with_ai_error(tmp_path: Path) -> None:
    sid = "SID-RUN-ERR"
    runs_root = tmp_path / "runs"
    accounts_dir = runs_root / sid / "cases" / "accounts"

    bureaus_payload = {
        "transunion": {
            "balance_owed": "500",
            "account_number": "654321",
            "date_opened": "2019-05-05",
        }
    }

    _write_account_payload(accounts_dir, 3, bureaus_payload)
    _write_account_payload(accounts_dir, 4, bureaus_payload)

    computation = score_accounts(sid, runs_root=runs_root, write_tags=True)

    ai_sender.write_error_tags(
        runs_root,
        sid,
        3,
        4,
        "Timeout",
        "request timed out",
        "2024-01-02T00:00:00Z",
    )

    outcomes = _load_ai_outcomes(sid, runs_root)
    rows = prepare_summary_rows(computation, outcomes)

    assert rows
    row = rows[0]
    assert row["idx"] in {3, 4}
    assert row["ai_with"] in {3, 4}
    assert row["ai_decision"] == "error:Timeout"
    assert row["ai_reason"] == "request timed out"
