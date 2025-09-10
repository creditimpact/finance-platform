import json
from pathlib import Path

from backend.core.logic.report_analysis.problem_case_builder import build_problem_cases


def test_problem_case_builder(tmp_path):
    """Smoke test for :func:`build_problem_cases`.

    The builder should read a sample ``accounts_from_full.json`` file and
    generate the expected output structure under ``cases/``.
    """

    sid = "S123"
    accounts = [
        {
            "account_index": 1,
            "heading_guess": None,
            "lines": [],
            "fields": {"past_due_amount": 20},
        },
        {"account_index": 2, "heading_guess": "OK", "lines": [], "fields": {}},
        {"account_index": 3, "heading_guess": "OK", "lines": [], "fields": {}},
    ]

    # Write sample accounts_from_full.json
    acc_path = (
        tmp_path
        / "traces"
        / "blocks"
        / sid
        / "accounts_table"
        / "accounts_from_full.json"
    )
    acc_path.parent.mkdir(parents=True, exist_ok=True)
    acc_path.write_text(json.dumps({"accounts": accounts}), encoding="utf-8")

    # Run the builder
    summary = build_problem_cases(sid, root=tmp_path)

    # Validate index.json was written with correct counts
    index_path = tmp_path / "cases" / sid / "index.json"
    assert index_path.exists()
    index = json.loads(index_path.read_text())
    assert index["total"] == len(accounts)
    assert index["problematic"] == 1

    # Ensure at least one account case file exists with problem details
    acc_dir = tmp_path / "cases" / sid / "accounts"
    files = list(acc_dir.glob("*.json"))
    assert files, "expected at least one account case file"
    case_data = json.loads(files[0].read_text())
    assert case_data.get("problem_tags") or case_data.get("problem_reasons")

    # Returned summary should mirror disk counts
    assert summary["total"] == len(accounts)
    assert summary["problematic"] == 1
