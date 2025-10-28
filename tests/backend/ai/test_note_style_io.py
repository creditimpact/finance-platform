import json
from pathlib import Path

from backend.ai.note_style.io import note_style_snapshot
from backend.core.ai.paths import (
    ensure_note_style_paths,
    note_style_pack_filename,
    note_style_result_filename,
    normalize_note_style_account_id,
)


def _write_index(paths, accounts: list[str]) -> None:
    packs = []
    for account in accounts:
        packs.append(
            {
                "account_id": account,
                "pack_path": f"packs/{note_style_pack_filename(account)}",
                "status": "built",
            }
        )
    payload = {"packs": packs, "totals": {"packs_total": len(packs)}}
    paths.index_file.parent.mkdir(parents=True, exist_ok=True)
    paths.index_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_pack(paths, account: str) -> Path:
    pack_path = paths.packs_dir / note_style_pack_filename(account)
    pack_path.parent.mkdir(parents=True, exist_ok=True)
    pack_path.write_text("{}\n", encoding="utf-8")
    return pack_path


def _write_result(paths, account: str, payload: dict[str, object]) -> Path:
    result_path = paths.results_dir / note_style_result_filename(account)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")
    return result_path


def test_note_style_snapshot_empty(tmp_path: Path) -> None:
    snapshot = note_style_snapshot("SID-EMPTY", runs_root=tmp_path)
    assert snapshot.packs_expected == set()
    assert snapshot.packs_built == set()
    assert snapshot.packs_completed == set()
    assert snapshot.packs_failed == set()


def test_note_style_snapshot_derives_expected_from_packs(tmp_path: Path) -> None:
    sid = "SID-PACKS"
    accounts = ["A1", "A2", "A3"]
    paths = ensure_note_style_paths(tmp_path, sid, create=True)
    _write_index(paths, accounts)

    _write_pack(paths, accounts[0])
    _write_pack(paths, accounts[2])

    snapshot = note_style_snapshot(sid, runs_root=tmp_path)
    expected = {normalize_note_style_account_id(account) for account in accounts}
    assert snapshot.packs_expected == expected

    built_expected = {
        normalize_note_style_account_id(accounts[0]),
        normalize_note_style_account_id(accounts[2]),
    }
    assert snapshot.packs_built == built_expected


def test_note_style_snapshot_missing_index(tmp_path: Path) -> None:
    sid = "SID-NO-INDEX"
    paths = ensure_note_style_paths(tmp_path, sid, create=True)
    account = "B1"
    _write_pack(paths, account)

    if paths.index_file.exists():
        paths.index_file.unlink()

    snapshot = note_style_snapshot(sid, runs_root=tmp_path)
    normalized = normalize_note_style_account_id(account)
    assert snapshot.packs_expected == {normalized}
    assert snapshot.packs_built == {normalized}


def test_note_style_snapshot_tracks_results(tmp_path: Path) -> None:
    sid = "SID-RESULTS"
    accounts = ["A1", "A2", "A3"]
    paths = ensure_note_style_paths(tmp_path, sid, create=True)
    _write_index(paths, accounts)
    for account in accounts:
        _write_pack(paths, account)

    snapshot = note_style_snapshot(sid, runs_root=tmp_path)
    assert snapshot.packs_completed == set()
    assert snapshot.packs_failed == set()

    _write_result(paths, accounts[0], {"analysis": {"tone": "neutral"}})
    snapshot = note_style_snapshot(sid, runs_root=tmp_path)
    assert snapshot.packs_completed == {normalize_note_style_account_id(accounts[0])}
    assert snapshot.packs_failed == set()

    _write_result(paths, accounts[1], {"status": "failed", "error": {"reason": "timeout"}})
    snapshot = note_style_snapshot(sid, runs_root=tmp_path)
    assert snapshot.packs_completed == {normalize_note_style_account_id(accounts[0])}
    assert snapshot.packs_failed == {normalize_note_style_account_id(accounts[1])}

    _write_result(paths, accounts[2], {"error": "invalid_result"})
    snapshot = note_style_snapshot(sid, runs_root=tmp_path)
    assert snapshot.packs_failed == {
        normalize_note_style_account_id(accounts[1]),
        normalize_note_style_account_id(accounts[2]),
    }
    assert snapshot.packs_completed == {normalize_note_style_account_id(accounts[0])}
