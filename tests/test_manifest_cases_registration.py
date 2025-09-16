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

