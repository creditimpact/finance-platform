import json
from pathlib import Path

import pytest

from backend.ai.note_style.cli import diagnose_note_style_stage, main

from tests.ai.note_style._helpers import prime_stage


def _write_runflow(path: Path, status: str) -> None:
    payload = {"stages": {"note_style": {"status": status, "sent": status == "success"}}}
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def test_diagnose_recommends_send(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    sid = "SID-DIAG-SEND"
    accounts = ["idx-001", "idx-002"]

    prime_stage(
        tmp_path,
        sid,
        expected_accounts=accounts,
        built_accounts=accounts,
    )

    run_dir = tmp_path / sid
    _write_runflow(run_dir / "runflow.json", "success")

    exit_code = main(["diagnose", "--sid", sid, "--runs-root", str(tmp_path)])
    assert exit_code == 0

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert payload["state"]["status"] == "built"
    assert payload["runflow"]["status"] == "success"
    assert payload["runflow"]["mismatch"] is True
    recommended = payload["recommended_action"]
    assert recommended["action"] == "send"
    assert set(recommended["accounts"]) == set(accounts)
    assert recommended["count"] == len(accounts)


def test_diagnose_reports_completion(tmp_path: Path) -> None:
    sid = "SID-DIAG-COMPLETE"
    accounts = ["idx-100", "idx-101", "idx-102"]

    prime_stage(
        tmp_path,
        sid,
        expected_accounts=accounts,
        built_accounts=accounts,
        completed_accounts=accounts,
    )

    result = diagnose_note_style_stage(sid, runs_root=tmp_path)

    assert result["state"]["status"] == "success"
    assert result["recommended_action"]["action"] == "complete"
    assert result["counts"]["completed"] == len(accounts)
    assert result["runflow"]["mismatch"] is False
