from __future__ import annotations

from pathlib import Path

from backend.core.ai.paths import (
    NoteStyleAccountPaths,
    ensure_note_style_account_paths,
    ensure_note_style_paths,
    normalize_note_style_account_id,
    note_style_pack_filename,
    note_style_result_filename,
)


def test_ensure_note_style_paths_creates_directories(tmp_path: Path) -> None:
    sid = "SID123"
    paths = ensure_note_style_paths(tmp_path, sid, create=True)

    base = (tmp_path / sid / "ai_packs" / "note_style").resolve()
    assert paths.base == base
    assert paths.packs_dir == (base / "packs").resolve()
    assert paths.results_dir == (base / "results").resolve()
    assert paths.debug_dir == (base / "debug").resolve()
    assert paths.index_file == (base / "index.json").resolve()
    assert paths.log_file == (base / "logs.txt").resolve()
    assert paths.packs_dir.is_dir()
    assert paths.results_dir.is_dir()
    assert paths.debug_dir.is_dir()


def test_ensure_note_style_paths_read_only(tmp_path: Path) -> None:
    sid = "SID456"
    base = tmp_path / sid / "ai_packs" / "note_style"

    paths = ensure_note_style_paths(tmp_path, sid, create=False)

    assert paths.base == base.resolve()
    assert not base.exists()
    assert not (base / "packs").exists()
    assert not (base / "results").exists()
    assert paths.debug_dir == (base / "debug").resolve()
    assert paths.log_file == (base / "logs.txt").resolve()


def test_note_style_filename_sanitizes_account_id() -> None:
    account_id = " idx/Account 42 "
    assert note_style_pack_filename(account_id) == "style_acc_idx_Account_42.jsonl"
    assert (
        note_style_result_filename(account_id)
        == "acc_idx_Account_42.result.jsonl"
    )


def test_note_style_account_paths_match_expected(tmp_path: Path) -> None:
    paths = ensure_note_style_paths(tmp_path, "SID789", create=True)
    account_paths = ensure_note_style_account_paths(paths, "idx-001", create=True)

    assert isinstance(account_paths, NoteStyleAccountPaths)
    expected_pack = paths.packs_dir / "style_acc_idx-001.jsonl"
    expected_result = paths.results_dir / "acc_idx-001.result.jsonl"
    expected_debug = paths.debug_dir / "idx-001.context.json"

    assert account_paths.account_id == "idx-001"
    assert account_paths.pack_file == expected_pack
    assert account_paths.result_file == expected_result
    assert account_paths.debug_file == expected_debug
    assert account_paths.pack_file.parent.is_dir()
    assert account_paths.result_file.parent.is_dir()
    assert account_paths.debug_file.parent.is_dir()


def test_note_style_filename_defaults_to_account_when_empty() -> None:
    assert note_style_pack_filename("") == "style_acc_account.jsonl"
    assert note_style_result_filename(None) == "acc_account.result.jsonl"


def test_normalize_note_style_account_id_matches_filename_normalization() -> None:
    account_id = " Account/ID 007 "
    normalized = normalize_note_style_account_id(account_id)

    assert normalized == "Account_ID_007"
    assert note_style_pack_filename(account_id) == f"style_acc_{normalized}.jsonl"
    assert note_style_result_filename(account_id) == f"acc_{normalized}.result.jsonl"
