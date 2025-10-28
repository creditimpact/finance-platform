from __future__ import annotations

from pathlib import Path

from ._helpers import prime_stage, stage_view


def test_stage_view_empty_when_no_packs(tmp_path: Path) -> None:
    sid = "SID-EMPTY"

    view = stage_view(tmp_path, sid)

    assert view.state == "empty"
    assert view.total_expected == 0
    assert not view.ready_to_send
    assert view.is_terminal is True


def test_stage_view_requires_all_packs_for_built(tmp_path: Path) -> None:
    sid = "SID-BUILT"
    accounts = ["idx-001", "idx-002", "idx-003"]

    prime_stage(
        tmp_path,
        sid,
        expected_accounts=accounts,
        built_accounts=[accounts[0]],
    )

    partial_view = stage_view(tmp_path, sid)
    assert partial_view.state == "pending"
    assert partial_view.built_total == 1
    assert partial_view.ready_to_send == frozenset({accounts[0]})

    prime_stage(
        tmp_path,
        sid,
        expected_accounts=accounts,
        built_accounts=accounts,
    )

    built_view = stage_view(tmp_path, sid)
    assert built_view.state == "built"
    assert built_view.built_total == len(accounts)
    assert built_view.ready_to_send == frozenset(accounts)


def test_stage_view_in_progress_and_success_states(tmp_path: Path) -> None:
    sid = "SID-STATES"
    accounts = ["idx-010", "idx-011", "idx-012"]

    prime_stage(
        tmp_path,
        sid,
        expected_accounts=accounts,
        built_accounts=accounts,
        completed_accounts=[accounts[0]],
    )

    in_progress_view = stage_view(tmp_path, sid)
    assert in_progress_view.state == "in_progress"
    assert in_progress_view.terminal_total == 1
    assert in_progress_view.ready_to_send == frozenset(accounts[1:])
    assert in_progress_view.is_terminal is False

    prime_stage(
        tmp_path,
        sid,
        expected_accounts=accounts,
        built_accounts=accounts,
        completed_accounts=accounts[:2],
        failed_accounts=accounts[2:],
    )

    success_view = stage_view(tmp_path, sid)
    assert success_view.state == "success"
    assert success_view.terminal_total == len(accounts)
    assert success_view.ready_to_send == frozenset()
    assert success_view.is_terminal is True

