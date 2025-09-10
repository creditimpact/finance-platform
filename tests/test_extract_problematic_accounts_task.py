import json
import logging
from pathlib import Path

import backend.api.tasks as task_module
from backend.api.tasks import extract_problematic_accounts
from backend.core.case_store import storage as cs_storage
from backend.core.logic.report_analysis import problem_case_builder
from backend import settings


def _write_accounts(base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    data = {"accounts": [{"account_index": 1, "fields": {"past_due_amount": 50}}]}
    base.write_text(json.dumps(data), encoding="utf-8")


def test_extract_problematic_accounts_task_builder(tmp_path, monkeypatch, caplog):
    sid = "S777"

    # Redirect project root and case store to temporary paths
    monkeypatch.setattr(settings, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(problem_case_builder, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(task_module, "PROJECT_ROOT", tmp_path, raising=False)

    casestore_dir = tmp_path / "casestore"
    monkeypatch.setattr("backend.config.CASESTORE_DIR", str(casestore_dir), raising=False)
    monkeypatch.setattr(cs_storage, "CASESTORE_DIR", str(casestore_dir))

    # Create Stage-A account artifacts
    acc_path = (
        tmp_path
        / "traces"
        / "blocks"
        / sid
        / "accounts_table"
        / "accounts_from_full.json"
    )
    _write_accounts(acc_path)

    caplog.set_level(logging.INFO)
    result = extract_problematic_accounts.run(sid)

    assert result["sid"] == sid
    assert len(result["found"]) == 1
    assert result["found"][0]["account_id"] == "account_1"
    assert any(f"PROBLEMATIC start sid={sid}" in m for m in caplog.messages)
    assert any(f"PROBLEMATIC done sid={sid} found=1" in m for m in caplog.messages)
