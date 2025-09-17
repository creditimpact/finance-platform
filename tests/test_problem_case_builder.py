import json
import logging
import os
from pathlib import Path

import pytest

from backend.core.logic.report_analysis.problem_case_builder import build_problem_cases
from backend.core.logic.report_analysis.account_merge import (
    DEFAULT_CFG,
    cluster_problematic_accounts,
)
from backend.core.logic.report_analysis.problem_extractor import (
    detect_problem_accounts,
    load_stagea_accounts_from_manifest,
)
from backend.pipeline.runs import RUNS_ROOT_ENV, RunManifest


@pytest.fixture
def tmp_sid_fixture(tmp_path, monkeypatch):
    sid = "SLEAN"
    runs_root = tmp_path / "runs"
    monkeypatch.setenv(RUNS_ROOT_ENV, str(runs_root))
    monkeypatch.setenv("CASES_LEAN_MODE", "1")

    m = RunManifest.for_sid(sid)
    traces_dir = m.ensure_run_subdir("traces_dir", "traces")
    acct_dir = traces_dir / "accounts_table"
    acct_dir.mkdir(parents=True, exist_ok=True)

    accounts = [
        {
            "account_id": "acct-001",
            "account_index": 1,
            "heading_guess": "Charge Off",
            "page_start": 2,
            "line_start": 5,
            "page_end": 2,
            "line_end": 6,
            "account_number": "1234 5678 90",
            "date_opened": "01-01-2020",
            "date_of_last_activity": "01-03-2023",
            "lines": [
                {"page": 2, "line": 5, "text": "Account 1 line 1"},
                {"page": 2, "line": 6, "text": "Account 1 line 2"},
            ],
            "fields": {
                "past_due_amount": 150.0,
                "balance_owed": 1200.0,
                "payment_status": "Charge Off",
                "account_status": "Closed",
            },
            "triad_fields": {
                "transunion": {
                    "past_due_amount": "150",
                    "balance_owed": "1200",
                    "payment_status": "Charge Off",
                    "account_status": "Closed",
                    "triad_rows": [
                        {"label": "Balance Owed", "values": {"transunion": "1200"}}
                    ],
                },
                "experian": {
                    "past_due_amount": "0",
                    "balance_owed": "1180",
                    "payment_status": "Delinquent",
                    "account_status": "Closed",
                },
            },
            "triad_rows": [
                {
                    "triad_row": True,
                    "label": "Account #",
                    "key": "account_number_display",
                    "values": {
                        "transunion": "1234 5678 90",
                        "experian": "1234 5678 90",
                        "equifax": "1234 5678 90",
                    },
                    "last_bureau_with_text": "transunion",
                    "expect_values_on_next_line": False,
                }
            ],
            "triad": {"order": ["transunion", "experian", "equifax"]},
            "seven_year_history": {
                "transunion": {"late30": 1, "late60": 0, "late90": 0},
                "experian": {"late30": 0, "late60": 0, "late90": 0},
                "equifax": {"late30": 0, "late60": 0, "late90": 0},
            },
            "two_year_payment_history": {
                "transunion": ["30", "OK"],
                "experian": ["OK", "OK"],
                "equifax": ["OK", "OK"],
            },
        },
        {
            "account_id": "acct-002",
            "account_index": 2,
            "heading_guess": "Neutral Account",
            "lines": [],
            "fields": {
                "past_due_amount": 0.0,
                "balance_owed": 0.0,
                "payment_status": "Pays as Agreed",
                "account_status": "Open",
            },
            "triad_fields": {"transunion": {}},
        },
    ]

    acc_path = acct_dir / "accounts_from_full.json"
    gen_path = acct_dir / "general_info_from_full.json"
    acc_path.write_text(json.dumps({"accounts": accounts}), encoding="utf-8")
    gen_path.write_text(json.dumps({"client_name": "Lean Fixture"}), encoding="utf-8")
    m.set_artifact("traces.accounts_table", "accounts_json", acc_path)
    m.set_artifact("traces.accounts_table", "general_json", gen_path)

    return sid


def test_lean_cases_structure(tmp_sid_fixture):
    sid = tmp_sid_fixture

    candidates = detect_problem_accounts(sid)
    assert candidates, "expected detection to flag at least one account"

    candidates_found = cluster_problematic_accounts(candidates, DEFAULT_CFG, sid=sid)
    summary = build_problem_cases(sid, candidates_found)

    runs_root = Path(os.environ[RUNS_ROOT_ENV])
    adir = runs_root / sid / "cases" / "accounts"
    assert adir.exists(), "cases/accounts directory not created"

    acc_dir = next(p for p in adir.iterdir() if p.is_dir())
    for name in [
        "meta.json",
        "summary.json",
        "bureaus.json",
        "fields_flat.json",
        "raw_lines.json",
        "tags.json",
    ]:
        assert (acc_dir / name).exists(), f"missing expected file {name}"

    summary_text = (acc_dir / "summary.json").read_text(encoding="utf-8")
    assert "triad_rows" not in summary_text
    summary_data = json.loads(summary_text)
    assert summary_data.get("problem_reasons"), "summary missing reasons"
    assert summary_data.get("merge_tag"), "summary missing merge tag"
    assert summary_data.get("pointers"), "summary missing pointers"

    full_accounts = load_stagea_accounts_from_manifest(sid)
    by_idx = {acc["account_index"]: acc for acc in full_accounts}
    idx = int(acc_dir.name)
    bureaus_path = acc_dir / "bureaus.json"
    bureaus_data = json.loads(bureaus_path.read_text(encoding="utf-8"))
    expected_bureaus = {}
    for bureau, payload in (by_idx[idx]["triad_fields"] or {}).items():
        if isinstance(payload, dict):
            payload = {k: v for k, v in payload.items() if k != "triad_rows"}
        expected_bureaus[bureau] = payload

    assert bureaus_data == expected_bureaus
    for payload in bureaus_data.values():
        if isinstance(payload, dict):
            assert "triad_rows" not in payload

    assert summary["problematic"] == len(candidates_found)


def test_problem_case_builder(tmp_path, caplog, monkeypatch):
    """Smoke test for :func:`build_problem_cases`.

    The builder should read a sample ``accounts_from_full.json`` file and
    generate the expected output structure under ``runs/<sid>/cases/``.
    """

    sid = "S123"
    accounts = [
        {
            "account_id": "idx-001",
            "account_index": 1,
            "heading_guess": None,
            "page_start": 1,
            "line_start": 2,
            "page_end": 2,
            "line_end": 4,
            "lines": [
                {"page": 1, "line": 2, "text": "Account line 1"},
                {"page": 2, "line": 4, "text": "Account line 2"},
            ],
            "fields": {"past_due_amount": 20},
            "triad_fields": {
                "transunion": {
                    "past_due_amount": "30",
                    "balance_owed": "100",
                    "payment_status": "Late",
                }
            },
            "triad_rows": [
                {
                    "triad_row": True,
                    "label": "Account #",
                    "key": "account_number_display",
                    "values": {
                        "transunion": "12345",
                        "experian": "12345",
                        "equifax": "12345",
                    },
                    "last_bureau_with_text": "transunion",
                    "expect_values_on_next_line": False,
                }
            ],
        },
        {
            "account_index": 2,
            "heading_guess": "OK",
            "lines": [],
            "fields": {},
            "triad_fields": {},
        },
        {
            "account_index": 3,
            "heading_guess": "OK",
            "lines": [],
            "fields": {},
            "triad_fields": {},
        },
    ]

    # Configure runs root and manifest; write canonical Stage-A artifacts
    runs_root = tmp_path / "runs"
    monkeypatch.setenv(RUNS_ROOT_ENV, str(runs_root))
    m = RunManifest.for_sid(sid)
    traces_dir = m.ensure_run_subdir("traces_dir", "traces")
    acct_dir = traces_dir / "accounts_table"
    acct_dir.mkdir(parents=True, exist_ok=True)
    acc_path = acct_dir / "accounts_from_full.json"
    gen_path = acct_dir / "general_info_from_full.json"
    acc_path.write_text(json.dumps({"accounts": accounts}), encoding="utf-8")
    gen_path.write_text(json.dumps({"client_name": "Tester"}), encoding="utf-8")
    m.set_artifact("traces.accounts_table", "accounts_json", acc_path)
    m.set_artifact("traces.accounts_table", "general_json", gen_path)

    runs_root = tmp_path / "runs"
    monkeypatch.setenv(RUNS_ROOT_ENV, str(runs_root))

    # Run the builder
    caplog.set_level(logging.INFO)
    candidates = [
        {
            "account_id": "idx-001",
            "account_index": 1,
            "problem_tags": ["past_due"],
            "problem_reasons": ["past_due_amount"],
            "merge_tag": {
                "group_id": "g1",
                "decision": "auto",
                "score_to": [
                    {"account_index": 2, "score": 0.83, "decision": "auto"}
                ],
                "best_match": {
                    "account_index": 2,
                    "score": 0.83,
                    "decision": "auto",
                },
                "parts": {
                    "acct_num": 1.0,
                    "balance": 0.95,
                    "dates": 0.9,
                    "status": 1.0,
                    "strings": 0.7,
                },
            },
        }
    ]
    summary = build_problem_cases(sid, candidates=candidates)

    cases_dir = runs_root / sid / "cases"

    # Validate index.json was written with correct counts
    index_path = cases_dir / "index.json"
    assert index_path.exists()
    index = json.loads(index_path.read_text())
    assert index["total"] == len(accounts)
    assert index["problematic"] == 1

    # Ensure at least one account case file exists with problem details
    acc_dir = cases_dir / "accounts"
    case_dirs = [p for p in acc_dir.iterdir() if p.is_dir()]
    assert case_dirs, "expected at least one account case folder"
    first = acc_dir / "1"
    assert first.exists()
    assert (first / "meta.json").exists()
    assert (first / "raw_lines.json").exists()
    assert (first / "bureaus.json").exists()
    assert (first / "fields_flat.json").exists()
    assert (first / "tags.json").exists()
    assert (first / "summary.json").exists()

    meta = json.loads((first / "meta.json").read_text())
    assert meta["account_index"] == 1
    assert meta["heading_guess"] is None
    assert meta["pointers"]["raw"] == "raw_lines.json"

    raw_lines = json.loads((first / "raw_lines.json").read_text())
    assert raw_lines == accounts[0]["lines"]

    bureaus = json.loads((first / "bureaus.json").read_text())
    assert bureaus == accounts[0]["triad_fields"]

    fields_flat = json.loads((first / "fields_flat.json").read_text())
    assert "past_due_amount" in fields_flat
    assert all(k == k.lower() for k in fields_flat.keys())

    tags_data = json.loads((first / "tags.json").read_text())
    assert tags_data == []

    case_data = json.loads((first / "summary.json").read_text())
    assert case_data.get("problem_tags") or case_data.get("problem_reasons")
    assert case_data.get("merge_tag")
    assert case_data["merge_tag"]["decision"] == "auto"
    assert case_data["merge_tag"]["best_match"]["account_index"] == 2
    assert case_data["pointers"]["bureaus"] == "bureaus.json"
    assert "triad_rows" not in json.dumps(case_data)

    for json_path in first.glob("*.json"):
        assert "triad_rows" not in json_path.read_text(), json_path

    accounts_index_path = acc_dir / "index.json"
    acc_index = json.loads(accounts_index_path.read_text())
    item = next(i for i in acc_index["items"] if i["id"] == "1")
    assert item["merge_group_id"] == "g1"

    # Returned summary should mirror disk counts
    assert summary["total"] == len(accounts)
    assert summary["problematic"] == 1

    assert any(
        f"PROBLEM_CASES start sid={sid} total={len(accounts)} out={cases_dir}" in m
        for m in caplog.messages
    )
    assert any(
        f"PROBLEM_CASES done sid={sid} total={len(accounts)} problematic=1 out={cases_dir}"
        in m
        for m in caplog.messages
    )

    # Run manifest registration and breadcrumb
    m = RunManifest.for_sid(sid)
    assert m.data["base_dirs"]["cases_accounts_dir"] == str((cases_dir / "accounts").resolve())
    assert (Path(m.get("cases", "accounts_index")).is_absolute())
    assert (Path(m.get("cases", "problematic_ids")).is_absolute())
    breadcrumb = cases_dir / ".manifest"
    assert breadcrumb.read_text() == str(m.path.resolve())


def test_problem_case_builder_updates_merge_tag_only_for_existing_cases(
    tmp_path, monkeypatch
):
    sid = "S777"
    accounts = [
        {"account_index": 1, "heading_guess": "A", "lines": [], "fields": {}}
    ]

    runs_root = tmp_path / "runs"
    monkeypatch.setenv(RUNS_ROOT_ENV, str(runs_root))
    m = RunManifest.for_sid(sid)
    traces_dir = m.ensure_run_subdir("traces_dir", "traces")
    acct_dir = traces_dir / "accounts_table"
    acct_dir.mkdir(parents=True, exist_ok=True)
    acc_path = acct_dir / "accounts_from_full.json"
    gen_path = acct_dir / "general_info_from_full.json"
    acc_path.write_text(json.dumps({"accounts": accounts}), encoding="utf-8")
    gen_path.write_text(json.dumps({"client": "Tester"}), encoding="utf-8")
    m.set_artifact("traces.accounts_table", "accounts_json", acc_path)
    m.set_artifact("traces.accounts_table", "general_json", gen_path)

    first_candidates = [
        {
            "account_id": "idx-001",
            "account_index": 1,
            "problem_tags": ["initial-tag"],
            "problem_reasons": ["initial-reason"],
            "merge_tag": {
                "group_id": "g1",
                "decision": "auto",
                "score_to": [],
                "best_match": None,
                "parts": {},
            },
        }
    ]

    build_problem_cases(sid, candidates=first_candidates)

    cases_dir = runs_root / sid / "cases"
    summary_path = cases_dir / "accounts" / "1" / "summary.json"
    summary_data = json.loads(summary_path.read_text())
    summary_data["problem_reasons"] = ["manual-reason"]
    summary_data["problem_tags"] = ["manual-tag"]
    summary_data["primary_issue"] = "custom-issue"
    summary_path.write_text(json.dumps(summary_data, indent=2), encoding="utf-8")

    second_candidates = [
        {
            "account_id": "idx-001",
            "account_index": 1,
            "problem_tags": ["new-tag"],
            "problem_reasons": ["new-reason"],
            "merge_tag": {
                "group_id": "g2",
                "decision": "ai",
                "score_to": [
                    {"account_index": 5, "score": 0.55, "decision": "ai"}
                ],
                "best_match": {
                    "account_index": 5,
                    "score": 0.55,
                    "decision": "ai",
                },
                "parts": {"acct_num": 0.7},
            },
        }
    ]

    build_problem_cases(sid, candidates=second_candidates)

    updated_data = json.loads(summary_path.read_text())
    assert updated_data["problem_reasons"] == ["manual-reason"]
    assert updated_data["problem_tags"] == ["manual-tag"]
    assert updated_data["primary_issue"] == "custom-issue"
    assert updated_data["merge_tag"]["decision"] == "ai"
    assert updated_data["merge_tag"]["group_id"] == "g2"
    assert updated_data["merge_tag"]["best_match"]["account_index"] == 5

    accounts_index = json.loads((cases_dir / "accounts" / "index.json").read_text())
    assert accounts_index["items"][0]["merge_group_id"] == "g2"


def test_problem_case_builder_manifest_missing_accounts_raises(tmp_path, monkeypatch):
    sid = "S999"
    runs_root = tmp_path / "runs"
    monkeypatch.setenv(RUNS_ROOT_ENV, str(runs_root))
    RunManifest.for_sid(sid)  # create manifest without traces entries
    try:
        build_problem_cases(sid, candidates=[])
        assert False, "expected RuntimeError when manifest lacks traces.accounts_table"
    except RuntimeError:
        pass




