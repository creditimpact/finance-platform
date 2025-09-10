from __future__ import annotations

from pathlib import Path

import pytest

from backend.core.logic.report_analysis.trace_cleanup import (
    purge_trace_except_artifacts,
)

REQUIRED = {
    "accounts_table/_debug_full.tsv",
    "accounts_table/accounts_from_full.json",
    "accounts_table/general_info_from_full.json",
}


def _setup(tmp_path: Path, sid: str) -> Path:
    base = tmp_path / "traces" / "blocks" / sid / "accounts_table"
    base.mkdir(parents=True)
    for name in REQUIRED:
        (tmp_path / "traces" / "blocks" / sid / name).write_text("x")
    return base.parent


def test_purge_trace_happy_path(tmp_path: Path) -> None:
    sid = "happy"
    base = _setup(tmp_path, sid)
    accounts = base / "accounts_table"
    (accounts / "extra.txt").write_text("1")
    (base / "root_extra.txt").write_text("1")
    subdir = base / "subdir"
    subdir.mkdir()
    (subdir / "file.txt").write_text("1")

    res = purge_trace_except_artifacts(sid, root=tmp_path, dry_run=True)
    assert (accounts / "extra.txt").exists()
    assert (base / "root_extra.txt").exists()
    assert (subdir / "file.txt").exists()

    assert set(res["kept"]) == REQUIRED
    assert res["deleted"] == []
    assert set(res["skipped"]) == {
        "accounts_table/extra.txt",
        "root_extra.txt",
        "subdir/file.txt",
        "subdir",
    }
    assert res["texts_deleted"] is False

    res2 = purge_trace_except_artifacts(sid, root=tmp_path, dry_run=False)
    assert not (accounts / "extra.txt").exists()
    assert not (base / "root_extra.txt").exists()
    assert not subdir.exists()
    assert set(res2["deleted"]) == {
        "accounts_table/extra.txt",
        "root_extra.txt",
        "subdir/file.txt",
        "subdir",
    }
    assert res2["skipped"] == []
    assert res2["texts_deleted"] is False


def test_missing_artifact(tmp_path: Path) -> None:
    sid = "missing"
    base = tmp_path / "traces" / "blocks" / sid / "accounts_table"
    base.mkdir(parents=True)
    # create only two artifacts
    (base / "_debug_full.tsv").write_text("1")
    (base / "accounts_from_full.json").write_text("1")
    extra = base / "extra.txt"
    extra.write_text("1")

    with pytest.raises(RuntimeError):
        purge_trace_except_artifacts(sid, root=tmp_path)
    assert extra.exists()


def test_session_folder_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        purge_trace_except_artifacts("nope", root=tmp_path)


def test_keep_extra(tmp_path: Path) -> None:
    sid = "extra"
    base = _setup(tmp_path, sid)
    accounts = base / "accounts_table"
    keep_file = accounts / "keep.json"
    keep_file.write_text("1")
    remove_file = accounts / "remove.txt"
    remove_file.write_text("1")

    res = purge_trace_except_artifacts(
        sid,
        root=tmp_path,
        keep_extra=["accounts_table/keep.json"],
        dry_run=False,
    )
    assert keep_file.exists()
    assert not remove_file.exists()
    assert "accounts_table/keep.json" in res["kept"]
    assert res["texts_deleted"] is False


def test_delete_texts_sid_dry_run(tmp_path: Path) -> None:
    sid = "sid1"
    _setup(tmp_path, sid)
    texts_dir = tmp_path / "traces" / "texts" / sid
    (texts_dir / "a" / "b").mkdir(parents=True)
    (texts_dir / "a" / "b" / "file.txt").write_text("1")

    res = purge_trace_except_artifacts(sid, root=tmp_path, dry_run=True)
    assert texts_dir.exists()
    assert res["texts_deleted"] is True
    assert f"texts/{sid}/**" in res["deleted"]


def test_delete_texts_sid_real(tmp_path: Path) -> None:
    sid = "sid2"
    _setup(tmp_path, sid)
    texts_dir = tmp_path / "traces" / "texts" / sid
    (texts_dir / "file.txt").parent.mkdir(parents=True, exist_ok=True)
    (texts_dir / "file.txt").write_text("1")

    res = purge_trace_except_artifacts(sid, root=tmp_path, dry_run=False)
    assert not texts_dir.exists()
    assert res["texts_deleted"] is True
    assert f"texts/{sid}/**" in res["deleted"]


def test_keep_texts_flag(tmp_path: Path) -> None:
    sid = "sid3"
    _setup(tmp_path, sid)
    texts_dir = tmp_path / "traces" / "texts" / sid
    texts_dir.mkdir(parents=True)

    res = purge_trace_except_artifacts(
        sid, root=tmp_path, dry_run=False, delete_texts_sid=False
    )
    assert texts_dir.exists()
    assert res["texts_deleted"] is False
    assert f"texts/{sid}/**" not in res["deleted"]


def test_cases_directory_untouched(tmp_path: Path) -> None:
    """Running cleanup must not remove any files under cases/<sid>."""
    sid = "case"
    _setup(tmp_path, sid)
    cases_file = tmp_path / "cases" / sid / "case.json"
    cases_file.parent.mkdir(parents=True)
    cases_file.write_text("1")

    purge_trace_except_artifacts(sid, root=tmp_path, dry_run=False)
    assert cases_file.exists()
