import json
import logging
from pathlib import Path

from backend.core.logic.report_analysis.keys import compute_logical_account_key
from backend.core.logic.report_analysis.problem_case_builder import build_problem_cases


def _write_accounts(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def test_build_problem_cases(tmp_path, caplog):
    sid = "S123"
    accounts = [
        {
            "account_index": 1,
            "heading_guess": None,
            "lines": [],
            "fields": {"past_due_amount": 20},
        },
        {"account_index": 2, "heading_guess": "OK", "lines": [], "fields": {}},
    ]
    acc_path = (
        tmp_path
        / "traces"
        / "blocks"
        / sid
        / "accounts_table"
        / "accounts_from_full.json"
    )
    _write_accounts(acc_path, {"accounts": accounts})

    caplog.set_level(logging.INFO)
    result = build_problem_cases(sid, root=tmp_path)

    assert result == {
        "sid": sid,
        "total": 2,
        "problematic": 1,
        "out_dir": str(tmp_path / "cases" / sid),
        "summaries": [
            {
                "account_id": "account_1",
                "problem_tags": ["past_due_amount"],
                "problem_reasons": ["past_due_amount: 20.00"],
            }
        ],
    }

    case_file = tmp_path / "cases" / sid / "accounts" / "account_1.json"
    assert case_file.exists()
    case = json.loads(case_file.read_text())
    assert case["sid"] == sid
    assert case["problem_tags"] == ["past_due_amount"]
    assert case.get("confidence") == 0.0

    index_file = tmp_path / "cases" / sid / "index.json"
    assert index_file.exists()
    index = json.loads(index_file.read_text())
    assert index["total"] == 2
    assert index["problematic"] == 1
    assert index["problematic_accounts"][0]["account_id"] == "account_1"

    out_dir = tmp_path / "cases" / sid
    assert any(f"PROBLEM_CASES start sid={sid}" in msg for msg in caplog.messages)
    assert any(
        f"PROBLEM_CASES done sid={sid} total=2 problematic=1 out={out_dir}" in msg
        for msg in caplog.messages
    )


def test_build_problem_cases_top_level_list(tmp_path):
    sid = "S234"
    accounts = [
        {"account_index": 1, "heading_guess": None, "fields": {"past_due_amount": 5}}
    ]
    acc_path = (
        tmp_path
        / "traces"
        / "blocks"
        / sid
        / "accounts_table"
        / "accounts_from_full.json"
    )
    _write_accounts(acc_path, accounts)

    res = build_problem_cases(sid, root=tmp_path)

    case_file = tmp_path / "cases" / sid / "accounts" / "account_1.json"
    assert case_file.exists()
    index = json.loads((tmp_path / "cases" / sid / "index.json").read_text())
    assert index["problematic"] == 1
    assert res["total"] == 1


def test_account_id_uses_logical_key(tmp_path):
    sid = "S345"
    account = {
        "account_index": 1,
        "heading_guess": None,
        "fields": {
            "issuer": "Real Bank",
            "account_last4": "1234",
            "account_type": "REVOLVING",
            "opened_date": "2020-01-01",
            "past_due_amount": 50,
        },
    }
    acc_path = (
        tmp_path
        / "traces"
        / "blocks"
        / sid
        / "accounts_table"
        / "accounts_from_full.json"
    )
    _write_accounts(acc_path, {"accounts": [account]})

    res = build_problem_cases(sid, root=tmp_path)

    expected = compute_logical_account_key(
        "Real Bank", "1234", "REVOLVING", "2020-01-01"
    )
    assert res["summaries"][0]["account_id"] == expected
    case_file = tmp_path / "cases" / sid / "accounts" / f"{expected}.json"
    assert case_file.exists()
