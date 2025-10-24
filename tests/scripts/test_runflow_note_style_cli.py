from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import pytest

from backend.ai.note_style_results import store_note_style_result
from backend.core.ai.paths import ensure_note_style_account_paths, ensure_note_style_paths
from scripts import runflow as runflow_cli


def _write_response(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _sample_result_payload() -> Mapping[str, object]:
    return {
        "analysis": {
            "tone": "empathetic",
            "context_hints": {
                "timeframe": {"month": "2024-01", "relative": "last_month"},
                "topic": "payment_dispute",
                "entities": {"creditor": "capital one", "amount": 120.0},
            },
            "emphasis": ["paid_already"],
            "confidence": 0.82,
            "risk_flags": ["follow_up"],
        },
        "note_metrics": {"char_len": 42, "word_len": 9},
    }


def test_runflow_note_style_build_command(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    sid = "SIDCLI1"
    account_id = "acct-cli"
    response_dir = tmp_path / sid / "frontend" / "review" / "responses"

    _write_response(
        response_dir / f"{account_id}.result.json",
        {
            "sid": sid,
            "account_id": account_id,
            "answers": {"explanation": "Please help, I already paid."},
        },
    )

    exit_code = runflow_cli.main(
        ["note-style", "build", "--sid", sid, "--runs-root", str(tmp_path)]
    )
    assert exit_code == 0

    out = capsys.readouterr().out.strip()
    assert out
    payload = json.loads(out)
    assert payload["sid"] == sid
    assert payload["processed_accounts"] == [account_id]
    assert payload["counts"]["packs_total"] == 1

    paths = ensure_note_style_paths(tmp_path, sid, create=False)
    account_paths = ensure_note_style_account_paths(paths, account_id, create=False)
    assert account_paths.pack_file.is_file()
    assert account_paths.result_file.is_file()
    assert account_paths.debug_file.is_file()


def test_runflow_note_style_refresh_command(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    sid = "SIDCLI2"
    account_id = "acct-refresh"
    response_dir = tmp_path / sid / "frontend" / "review" / "responses"

    _write_response(
        response_dir / f"{account_id}.result.json",
        {
            "sid": sid,
            "account_id": account_id,
            "answers": {"explanation": "The bank already fixed this."},
        },
    )

    build_exit = runflow_cli.main(
        ["note-style", "build", "--sid", sid, "--runs-root", str(tmp_path)]
    )
    assert build_exit == 0
    capsys.readouterr()

    paths = ensure_note_style_paths(tmp_path, sid, create=False)
    account_paths = ensure_note_style_account_paths(paths, account_id, create=False)

    store_note_style_result(
        sid,
        account_id,
        {
            **_sample_result_payload(),
            "sid": sid,
            "account_id": account_id,
        },
        runs_root=tmp_path,
    )

    exit_code = runflow_cli.main(
        ["note-style", "refresh", "--sid", sid, "--runs-root", str(tmp_path)]
    )
    assert exit_code == 0

    out = capsys.readouterr().out.strip()
    assert out
    payload = json.loads(out)
    assert payload["sid"] == sid
    assert payload["counts"]["packs_completed"] == 1
    assert payload["stage"]["status"] == "success"
    results_payload = payload["stage"].get("results", {})
    assert results_payload.get("completed") == 1
