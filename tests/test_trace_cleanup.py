from pathlib import Path

from backend.core.logic.report_analysis.trace_cleanup import purge_after_export

def test_purge_after_export_keeps_only_final_artifacts(tmp_path: Path) -> None:
    root = tmp_path
    sid = "test-sid"
    blocks = root / "traces" / "blocks" / sid
    texts = root / "traces" / "texts" / sid
    acct = blocks / "accounts_table"
    acct.mkdir(parents=True)
    texts.mkdir(parents=True)

    keep = ["_debug_full.tsv", "accounts_from_full.json", "general_info_from_full.json"]
    for name in keep:
        (acct / name).write_text("ok")

    (acct / "noise.txt").write_text("x")
    (blocks / "other.json").write_text("{}")
    (texts / "dump.txt").write_text("x")

    summary = purge_after_export(sid=sid, project_root=root)

    kept = {p.name for p in acct.iterdir()}
    assert kept == set(keep)
    assert {p.name for p in blocks.iterdir()} == {"accounts_table"}
    assert not texts.exists()
    assert isinstance(summary, dict)
