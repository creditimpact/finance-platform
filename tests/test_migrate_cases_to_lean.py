import json

from backend.core.logic.report_analysis.problem_case_builder import (
    _build_bureaus_payload_from_stagea,
)
from scripts.migrate_cases_to_lean import POINTERS, migrate


def test_migrate_cases_to_lean(tmp_path):
    run_dir = tmp_path / "runs" / "SID123"
    account_dir = run_dir / "cases" / "accounts" / "1"
    account_dir.mkdir(parents=True, exist_ok=True)

    account_data = {
        "account_index": 1,
        "account_id": "acct-1",
        "heading_guess": "Legacy Account",
        "page_start": 1,
        "line_start": 2,
        "page_end": 1,
        "line_end": 4,
        "lines": [
            {"page": 1, "line": 2, "text": "Account line 1"},
            {"page": 1, "line": 3, "text": "Account line 2"},
        ],
        "triad_fields": {
            "transunion": {
                "past_due_amount": "125.00",
                "payment_status": "Charge Off",
                "triad_rows": [{"label": "Balance", "values": {"transunion": "125"}}],
            },
            "experian": {"balance_owed": "500"},
            "equifax": {},
        },
        "triad_rows": [
            {"label": "Account #", "values": {"transunion": "1234"}},
            {"label": "Balance", "values": {"transunion": "500"}},
        ],
        "seven_year_history": {"transunion": {"late30": 1, "late60": 0, "late90": 0}},
        "two_year_payment_history": {"transunion": ["30", "OK"]},
        "triad": {"order": ["transunion", "experian", "equifax"]},
    }

    summary_data = {
        "account_index": 1,
        "problem_reasons": ["legacy-reason"],
        "merge_tag": {"group_id": "legacy", "decision": "manual"},
        "reason": {
            "primary_issue": "collection",
            "problem_tags": ["legacy-tag"],
        },
    }

    (account_dir / "account.json").write_text(
        json.dumps(account_data, indent=2), encoding="utf-8"
    )
    (account_dir / "summary.json").write_text(
        json.dumps(summary_data, indent=2), encoding="utf-8"
    )

    stats = migrate([run_dir])
    assert stats["processed"] == 1
    assert stats["migrated"] == 1

    assert not (account_dir / "account.json").exists()

    raw_lines = json.loads((account_dir / POINTERS["raw"]).read_text(encoding="utf-8"))
    assert raw_lines == account_data["lines"]

    bureaus = json.loads((account_dir / POINTERS["bureaus"]).read_text(encoding="utf-8"))
    assert "triad_rows" not in json.dumps(bureaus)
    assert bureaus == _build_bureaus_payload_from_stagea(account_data)
    assert bureaus["transunion"]["payment_status"] == "Charge Off"
    assert "two_year_payment_history" in bureaus
    assert "seven_year_history" in bureaus
    assert bureaus["two_year_payment_history"] == account_data[
        "two_year_payment_history"
    ]
    assert bureaus["seven_year_history"] == account_data["seven_year_history"]
    for key in ("transunion", "experian", "equifax"):
        assert key in bureaus and isinstance(bureaus[key], dict)

    flat_fields = json.loads((account_dir / POINTERS["flat"]).read_text(encoding="utf-8"))
    assert flat_fields["past_due_amount"] == 125.0
    assert flat_fields["balance_owed"] == 500.0

    meta = json.loads((account_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["account_index"] == 1
    assert meta["pointers"] == POINTERS

    tags = json.loads((account_dir / POINTERS["tags"]).read_text(encoding="utf-8"))
    assert tags == []

    summary = json.loads((account_dir / POINTERS["summary"]).read_text(encoding="utf-8"))
    assert summary["problem_reasons"] == ["legacy-reason"]
    assert summary["problem_tags"] == ["legacy-tag"]
    assert summary["merge_tag"]["group_id"] == "legacy"
    assert summary["primary_issue"] == "collection"
    assert summary["pointers"] == POINTERS

    for json_path in account_dir.glob("*.json"):
        text = json_path.read_text(encoding="utf-8")
        assert "triad_rows" not in text
