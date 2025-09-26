import json
from pathlib import Path

from backend.pipeline.runs import RUNS_ROOT_ENV, RunManifest
from backend.core.logic.report_analysis.problem_case_builder import build_problem_cases


def test_manifest_cases_registration(tmp_path, monkeypatch):
    sid = "T100"
    runs_root = tmp_path / "runs"
    monkeypatch.setenv(RUNS_ROOT_ENV, str(runs_root))

    # Prepare minimal Stage-A artifacts and register in manifest
    m = RunManifest.for_sid(sid)
    traces_dir = m.ensure_run_subdir("traces_dir", "traces")
    acct_dir = traces_dir / "accounts_table"
    acct_dir.mkdir(parents=True, exist_ok=True)
    acc = acct_dir / "accounts_from_full.json"
    gen = acct_dir / "general_info_from_full.json"
    acc.write_text(json.dumps({"accounts": [{"account_index": 1, "fields": {}}]}), encoding="utf-8")
    gen.write_text(json.dumps({"client_name": "Unit"}), encoding="utf-8")
    m.set_artifact("traces.accounts_table", "accounts_json", acc)
    m.set_artifact("traces.accounts_table", "general_json", gen)

    # Run builder with one candidate
    build_problem_cases(
        sid,
        candidates=[{"account_id": "idx-001", "account_index": 1, "problem_tags": ["t"], "problem_reasons": ["r"]}],
    )

    # Check manifest registrations
    m2 = RunManifest.for_sid(sid)
    cases_accounts_dir = Path(m2.data["base_dirs"]["cases_accounts_dir"])  # absolute
    assert cases_accounts_dir.is_absolute()
    acc_index = Path(m2.get("cases", "accounts_index"))
    prob_ids = Path(m2.get("cases", "problematic_ids"))
    assert acc_index.is_absolute() and prob_ids.is_absolute()
    # Per-account dir registration
    per = Path(m2.get("cases.accounts.idx-001", "dir"))
    assert per.is_absolute() and per.exists()


def test_build_problem_cases_writes_tags(tmp_path):
    sid = "PAIR-100"

    # Stage-A fallback artifacts
    stage_dir = tmp_path / "traces" / "blocks" / sid / "accounts_table"
    stage_dir.mkdir(parents=True, exist_ok=True)

    account_a = {
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
        "lines": [],
        "two_year_payment_history": {},
        "seven_year_history": {},
        "triad": {"order": ["transunion", "experian", "equifax"]},
    }
    account_b = {
        "account_index": 1,
        "account_id": "1",
        "triad_fields": {
            "experian": {
                "balance_owed": "150",
                "last_payment": "2024-01-05",
                "past_due_amount": "50",
                "high_balance": "500",
                "account_type": "Credit Card",
                "date_of_last_activity": "2024-01-02",
                "date_opened": "2020-01-01",
            }
        },
        "lines": [],
        "two_year_payment_history": {},
        "seven_year_history": {},
        "triad": {"order": ["transunion", "experian", "equifax"]},
    }

    accounts_payload = {"accounts": [account_a, account_b]}
    (stage_dir / "accounts_from_full.json").write_text(
        json.dumps(accounts_payload, ensure_ascii=False),
        encoding="utf-8",
    )

    summary = build_problem_cases(
        sid,
        candidates=[
            {"account_index": 0, "account_id": "0", "primary_issue": "collection"},
            {"account_index": 1, "account_id": "1", "primary_issue": "collection"},
        ],
        root=tmp_path,
    )

    run_dir = tmp_path / "runs" / sid
    tags_a = json.loads(
        (run_dir / "cases" / "accounts" / "0" / "tags.json").read_text(encoding="utf-8")
    )
    tags_b = json.loads(
        (run_dir / "cases" / "accounts" / "1" / "tags.json").read_text(encoding="utf-8")
    )

    issue_a = [tag for tag in tags_a if tag.get("kind") == "issue"][0]
    issue_b = [tag for tag in tags_b if tag.get("kind") == "issue"][0]
    assert issue_a["type"] == issue_b["type"] == "collection"
    assert "details" not in issue_a
    assert "details" not in issue_b

    pair_a = [tag for tag in tags_a if tag.get("kind") == "merge_pair"][0]
    pair_b = [tag for tag in tags_b if tag.get("kind") == "merge_pair"][0]
    assert pair_a["with"] == 1 and pair_b["with"] == 0
    assert pair_a["decision"] == pair_b["decision"] == "ai"

    best_a = [tag for tag in tags_a if tag.get("kind") == "merge_best"][0]
    best_b = [tag for tag in tags_b if tag.get("kind") == "merge_best"][0]
    assert best_a["with"] == 1 and best_b["with"] == 0
    assert best_a["decision"] == best_b["decision"] == "ai"

    summary_path = run_dir / "cases" / "accounts" / "0" / "summary.json"
    summary_data = json.loads(summary_path.read_text(encoding="utf-8"))
    assert "merge_tag" not in summary_data
    merge_scoring_summary = summary_data.get("merge_scoring")
    if isinstance(merge_scoring_summary, dict):
        assert "acctnum_level" in merge_scoring_summary

    assert summary["merge_scoring"]["scores"][0][1]["decision"] == "ai"
    assert summary["merge_scoring"]["best"][0]["partner_index"] == 1
