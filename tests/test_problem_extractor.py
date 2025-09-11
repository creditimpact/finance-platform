import logging
from pathlib import Path

from backend.core.logic.report_analysis import problem_extractor


def test_problem_extractor_missing_accounts(tmp_path, caplog, monkeypatch):
    sid = "S404"
    monkeypatch.setattr(problem_extractor, "PROJECT_ROOT", tmp_path)

    caplog.set_level(logging.INFO)
    results = problem_extractor.detect_problem_accounts(sid)

    assert results == []
    acc_path = (
        Path(tmp_path)
        / "traces"
        / "blocks"
        / sid
        / "accounts_table"
        / "accounts_from_full.json"
    )
    assert any(
        f"accounts_from_full.json missing sid={sid} path={acc_path}" in m
        for m in caplog.messages
    )
    assert any(f"PROBLEM_EXTRACT start sid={sid}" in m for m in caplog.messages)
    assert any(
        f"PROBLEM_EXTRACT done sid={sid} total=0 problematic=0" in m
        for m in caplog.messages
    )
