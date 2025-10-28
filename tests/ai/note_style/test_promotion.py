from __future__ import annotations

from pathlib import Path

from backend.runflow.decider import _apply_note_style_stage_promotion

from ._helpers import prime_stage


def _stage_payload(data: dict, key: str) -> dict:
    stages = data.setdefault("stages", {})
    payload = stages.get(key)
    assert isinstance(payload, dict)
    return payload


def test_promotion_tracks_built_and_in_progress(tmp_path: Path) -> None:
    sid = "SID-PROMO"
    accounts = ["idx-100", "idx-101", "idx-102"]
    data: dict = {"stages": {}}
    run_dir = tmp_path / sid

    prime_stage(
        tmp_path,
        sid,
        expected_accounts=accounts,
        built_accounts=[accounts[0]],
    )

    updated, promoted, log_context = _apply_note_style_stage_promotion(data, run_dir)
    assert updated is True
    assert promoted is False
    stage = _stage_payload(data, "note_style")
    assert stage["status"] == "pending"
    assert stage["empty_ok"] is False
    assert stage["metrics"]["packs_total"] == len(accounts)
    assert log_context == {"total": len(accounts), "completed": 0, "failed": 0}

    prime_stage(
        tmp_path,
        sid,
        expected_accounts=accounts,
        built_accounts=accounts,
    )

    updated, promoted, _ = _apply_note_style_stage_promotion(data, run_dir)
    assert updated is True
    assert promoted is False
    stage = _stage_payload(data, "note_style")
    assert stage["status"] == "built"
    assert stage["results"]["completed"] == 0
    assert stage["results"]["failed"] == 0

    prime_stage(
        tmp_path,
        sid,
        expected_accounts=accounts,
        built_accounts=accounts,
        completed_accounts=[accounts[0]],
    )

    updated, promoted, _ = _apply_note_style_stage_promotion(data, run_dir)
    assert updated is True
    assert promoted is False
    stage = _stage_payload(data, "note_style")
    assert stage["status"] == "in_progress"
    assert stage["results"]["completed"] == 1
    assert stage["sent"] is False


def test_promotion_reaches_success_only_when_all_terminal(tmp_path: Path) -> None:
    sid = "SID-PROMO-SUCCESS"
    accounts = ["idx-200", "idx-201"]
    data: dict = {"stages": {}}
    run_dir = tmp_path / sid

    prime_stage(
        tmp_path,
        sid,
        expected_accounts=accounts,
        built_accounts=accounts,
        completed_accounts=[accounts[0]],
    )

    _apply_note_style_stage_promotion(data, run_dir)
    stage = _stage_payload(data, "note_style")
    assert stage["status"] == "in_progress"
    assert stage["sent"] is False

    prime_stage(
        tmp_path,
        sid,
        expected_accounts=accounts,
        built_accounts=accounts,
        completed_accounts=accounts,
    )

    updated, promoted, log_context = _apply_note_style_stage_promotion(data, run_dir)
    assert updated is True
    assert promoted is True
    stage = _stage_payload(data, "note_style")
    assert stage["status"] == "success"
    assert stage["sent"] is True
    assert stage["results"]["completed"] == len(accounts)
    assert stage["results"]["failed"] == 0
    assert log_context == {"total": len(accounts), "completed": len(accounts), "failed": 0}


def test_promotion_marks_empty_when_no_accounts(tmp_path: Path) -> None:
    sid = "SID-PROMO-EMPTY"
    data: dict = {"stages": {}}
    run_dir = tmp_path / sid

    updated, promoted, log_context = _apply_note_style_stage_promotion(data, run_dir)

    assert updated is True
    assert promoted is True
    stage = _stage_payload(data, "note_style")
    assert stage["status"] == "empty"
    assert stage["empty_ok"] is True
    assert stage["sent"] is True
    assert log_context == {"total": 0, "completed": 0, "failed": 0}

