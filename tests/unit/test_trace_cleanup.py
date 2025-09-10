from __future__ import annotations

from pathlib import Path

import pytest

from backend.core.logic.report_analysis.trace_cleanup import purge_trace_except_artifacts


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
