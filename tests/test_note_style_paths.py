from __future__ import annotations

from pathlib import Path

import json
import pytest

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
    assert paths.results_raw_dir == (base / "results_raw").resolve()
    assert paths.debug_dir == (base / "debug").resolve()
    assert paths.index_file == (base / "index.json").resolve()
    assert paths.log_file == (base / "logs.txt").resolve()
    assert paths.packs_dir.is_dir()
    assert paths.results_dir.is_dir()
    assert paths.results_raw_dir.is_dir()
    assert paths.debug_dir.is_dir()


def test_ensure_note_style_paths_read_only(tmp_path: Path) -> None:
    sid = "SID456"
    base = tmp_path / sid / "ai_packs" / "note_style"

    paths = ensure_note_style_paths(tmp_path, sid, create=False)

    assert paths.base == base.resolve()
    assert not base.exists()
    assert not (base / "packs").exists()
    assert not (base / "results").exists()
    assert not (base / "results_raw").exists()
    assert paths.debug_dir == (base / "debug").resolve()
    assert paths.log_file == (base / "logs.txt").resolve()


def test_note_style_paths_use_manifest_when_enabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sid = "SID999"
    run_dir = tmp_path / sid
    run_dir.mkdir(parents=True, exist_ok=True)

    base_dir = tmp_path / "custom_base"
    packs_dir = base_dir / "packs_alt"
    results_dir = base_dir / "results_alt"
    index_path = base_dir / "custom_index.json"
    log_path = base_dir / "custom_logs.txt"

    manifest_payload = {
        "ai": {
            "packs": {
                "note_style": {
                    "base": str(base_dir),
                    "packs_dir": str(packs_dir),
                    "results_dir": str(results_dir),
                    "index": str(index_path),
                    "logs": str(log_path),
                }
            }
        }
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    monkeypatch.setattr("backend.config.NOTE_STYLE_USE_MANIFEST_PATHS", True, raising=False)

    paths = ensure_note_style_paths(tmp_path, sid, create=False)

    assert paths.base == base_dir.resolve()
    assert paths.packs_dir == packs_dir.resolve()
    assert paths.results_dir == results_dir.resolve()
    assert paths.index_file == index_path.resolve()
    assert paths.log_file == log_path.resolve()
    assert paths.results_raw_dir == (base_dir / "results_raw").resolve()
    assert paths.debug_dir == (base_dir / "debug").resolve()


def test_note_style_paths_normalize_windows_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sid = "SIDWIN"
    run_dir = tmp_path / sid
    run_dir.mkdir(parents=True, exist_ok=True)

    windows_base = f"C:\\author\\runs\\{sid}\\ai_packs\\note_style"
    windows_packs = windows_base + "\\packs"
    windows_results = windows_base + "\\results"
    windows_index = windows_base + "\\index.json"
    windows_logs = windows_base + "\\logs.txt"

    manifest_payload = {
        "ai": {
            "packs": {
                "note_style": {
                    "base": windows_base,
                    "packs_dir": windows_packs,
                    "results_dir": windows_results,
                    "index": windows_index,
                    "logs": windows_logs,
                }
            }
        }
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    monkeypatch.setattr("backend.config.NOTE_STYLE_USE_MANIFEST_PATHS", True, raising=False)

    paths = ensure_note_style_paths(tmp_path, sid, create=False)

    expected_base = (tmp_path / sid / "ai_packs" / "note_style").resolve()
    assert paths.base == expected_base
    assert paths.packs_dir == (expected_base / "packs").resolve()
    assert paths.results_dir == (expected_base / "results").resolve()
    assert paths.index_file == (expected_base / "index.json").resolve()
    assert paths.log_file == (expected_base / "logs.txt").resolve()


def test_note_style_filename_sanitizes_account_id() -> None:
    account_id = " idx/Account 42 "
    assert note_style_pack_filename(account_id) == "acc_idx_Account_42.jsonl"
    assert (
        note_style_result_filename(account_id)
        == "acc_idx_Account_42.result.jsonl"
    )


def test_note_style_account_paths_match_expected(tmp_path: Path) -> None:
    paths = ensure_note_style_paths(tmp_path, "SID789", create=True)
    account_paths = ensure_note_style_account_paths(paths, "idx-001", create=True)

    assert isinstance(account_paths, NoteStyleAccountPaths)
    expected_pack = paths.packs_dir / "acc_idx-001.jsonl"
    expected_result = paths.results_dir / "acc_idx-001.result.jsonl"
    expected_debug = paths.debug_dir / "idx-001.context.json"
    expected_raw = paths.results_raw_dir / "acc_idx-001.raw.txt"

    assert account_paths.account_id == "idx-001"
    assert account_paths.pack_file == expected_pack
    assert account_paths.result_file == expected_result
    assert account_paths.result_raw_file == expected_raw
    assert account_paths.debug_file == expected_debug
    assert account_paths.pack_file.parent.is_dir()
    assert account_paths.result_file.parent.is_dir()
    assert account_paths.result_raw_file.parent.is_dir()
    assert account_paths.debug_file.parent.is_dir()


def test_note_style_filename_defaults_to_account_when_empty() -> None:
    assert note_style_pack_filename("") == "acc_account.jsonl"
    assert note_style_result_filename(None) == "acc_account.result.jsonl"


def test_normalize_note_style_account_id_matches_filename_normalization() -> None:
    account_id = " Account/ID 007 "
    normalized = normalize_note_style_account_id(account_id)

    assert normalized == "Account_ID_007"
    assert note_style_pack_filename(account_id) == f"acc_{normalized}.jsonl"
    assert note_style_result_filename(account_id) == f"acc_{normalized}.result.jsonl"
