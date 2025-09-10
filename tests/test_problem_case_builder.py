import json
import logging
from pathlib import Path

from backend.core.logic.report_analysis.problem_case_builder import build_problem_cases


def _write_accounts(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def test_build_problem_cases(tmp_path, caplog):
    sid = "S123"
    accounts = [
        {"account_index": 1, "heading_guess": None, "lines": []},
        {"account_index": 2, "heading_guess": "OK", "lines": []},
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
                "account_id": "1",
                "problem_tags": ["missing_heading"],
                "problem_reasons": ["missing heading"],
            }
        ],
    }

    case_file = tmp_path / "cases" / sid / "accounts" / "1.json"
    assert case_file.exists()
    case = json.loads(case_file.read_text())
    assert case["sid"] == sid
    assert case["problem_tags"] == ["missing_heading"]

    index_file = tmp_path / "cases" / sid / "index.json"
    assert index_file.exists()
    index = json.loads(index_file.read_text())
    assert index["total"] == 2
    assert index["problematic"] == 1
    assert index["problematic_accounts"][0]["account_id"] == "1"

    assert any("PROBLEM_CASES start" in msg for msg in caplog.messages)
    assert any("PROBLEM_CASES done" in msg for msg in caplog.messages)


def test_build_problem_cases_top_level_list(tmp_path):
    sid = "S234"
    accounts = [{"account_index": 1, "heading_guess": None}]
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

    case_file = tmp_path / "cases" / sid / "accounts" / "1.json"
    assert case_file.exists()
    index = json.loads((tmp_path / "cases" / sid / "index.json").read_text())
    assert index["problematic"] == 1
    assert res["total"] == 1
