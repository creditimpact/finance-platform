from __future__ import annotations

from pathlib import Path

from backend.ai.validation_index import ValidationIndexEntry, ValidationPackIndexWriter
from backend.core.ai.paths import (
    ensure_validation_paths,
    validation_pack_filename_for_account,
    validation_result_jsonl_filename_for_account,
    validation_result_filename_for_account,
)

from devtools import show_validation_index


def _create_index_entry(
    validation_root: Path,
    account_id: int,
    *,
    status: str,
    weak_fields: list[str] | None = None,
    model: str | None = None,
    lines: int = 0,
) -> ValidationIndexEntry:
    packs_dir = validation_root / "packs"
    results_dir = validation_root / "results"

    pack_path = packs_dir / validation_pack_filename_for_account(account_id)
    summary_path = results_dir / validation_result_filename_for_account(account_id)
    jsonl_path = results_dir / validation_result_jsonl_filename_for_account(account_id)

    pack_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    pack_path.write_text("", encoding="utf-8")
    summary_path.write_text("{}\n", encoding="utf-8")
    jsonl_path.write_text("", encoding="utf-8")

    return ValidationIndexEntry(
        account_id=account_id,
        pack_path=pack_path,
        result_jsonl_path=jsonl_path,
        result_json_path=summary_path,
        weak_fields=weak_fields or [],
        line_count=lines or len(weak_fields or ()),
        status=status,
        model=model,
        request_lines=lines or len(weak_fields or ()),
        source_hash="hash",
    )


def test_show_validation_index_outputs_table(tmp_path, capsys):
    sid = "SID123"
    runs_root = tmp_path
    validation_paths = ensure_validation_paths(runs_root, sid, create=True)

    entry_ok = _create_index_entry(
        validation_paths.base,
        14,
        status="ok",
        weak_fields=["field_a", "field_b"],
        model="gpt-test",
    )
    entry_error = _create_index_entry(
        validation_paths.base,
        7,
        status="error",
        weak_fields=["history_2y"],
    )

    writer = ValidationPackIndexWriter(
        sid=sid,
        index_path=validation_paths.index_file,
        packs_dir=validation_paths.packs_dir,
        results_dir=validation_paths.results_dir,
    )
    writer.bulk_upsert([entry_ok, entry_error])

    exit_code = show_validation_index.main([sid, "--runs-root", str(runs_root)])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "SID: SID123" in captured.out
    assert "Index:" in captured.out
    # Table contents
    assert "014" in captured.out
    assert "007" in captured.out
    assert "gpt-test" in captured.out
    assert "ok" in captured.out
    assert "error" in captured.out
    assert "Status counts:" in captured.out
    assert "Accounts: 2  Weak fields: 3" in captured.out


def test_show_validation_index_handles_missing_index(tmp_path, capsys):
    sid = "MISSING"
    runs_root = tmp_path

    exit_code = show_validation_index.main([sid, "--runs-root", str(runs_root)])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "No validation packs recorded in the index." in captured.out
