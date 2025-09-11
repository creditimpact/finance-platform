import json
import logging

from backend.core.logic.report_analysis.problem_case_builder import build_problem_cases


def test_problem_case_builder(tmp_path, caplog):
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
    caplog.set_level(logging.INFO)
    candidates = [
        {
            "account_id": "idx-001",
            "account_index": 1,
            "problem_tags": ["past_due"],
            "problem_reasons": ["past_due_amount"],
        }
    ]
    summary = build_problem_cases(sid, candidates=candidates, root=tmp_path)

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

    out_dir = tmp_path / "cases" / sid
    assert any(
        f"PROBLEM_CASES start sid={sid} total={len(accounts)} out={out_dir}" in m
        for m in caplog.messages
    )
    assert any(
        f"PROBLEM_CASES done sid={sid} total={len(accounts)} problematic=1 out={out_dir}"
        in m
        for m in caplog.messages
    )


def test_problem_case_builder_missing_accounts(tmp_path, caplog):
    sid = "S999"

    caplog.set_level(logging.INFO)
    summary = build_problem_cases(sid, root=tmp_path)

    # Index should be written with zero counts
    index_path = tmp_path / "cases" / sid / "index.json"
    assert index_path.exists()
    index = json.loads(index_path.read_text())
    assert index["total"] == 0 and index["problematic"] == 0
    assert summary["total"] == 0 and summary["problematic"] == 0

    out_dir = tmp_path / "cases" / sid
    assert any(
        f"PROBLEM_CASES start sid={sid} total=0 out={out_dir}" in m
        for m in caplog.messages
    )
    assert any(
        f"PROBLEM_CASES done sid={sid} total=0 problematic=0 out={out_dir}" in m
        for m in caplog.messages
    )
