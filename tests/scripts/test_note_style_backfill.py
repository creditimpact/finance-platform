from __future__ import annotations

import json
from pathlib import Path

from scripts.note_style_backfill import backfill_note_style_runflow


def _write_runflow(path: Path, status: str) -> None:
    payload = {
        "stages": {
            "note_style": {
                "status": status,
                "empty_ok": False,
                "metrics": {"packs_total": 1},
                "results": {"results_total": 1, "completed": 1, "failed": 0},
                "summary": {
                    "packs_total": 1,
                    "results_total": 1,
                    "completed": 1,
                    "failed": 0,
                    "empty_ok": False,
                    "metrics": {"packs_total": 1},
                    "results": {"results_total": 1, "completed": 1, "failed": 0},
                },
                "sent": True,
                "completed_at": "2024-01-01T00:00:00Z",
                "last_at": "2024-01-01T00:00:00Z",
            }
        }
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_pack(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"account_id": "idx-001", "messages": []}) + "\n",
        encoding="utf-8",
    )


def test_backfill_downgrades_terminal_status(tmp_path: Path) -> None:
    sid = "SID001"
    run_dir = tmp_path / sid
    packs_dir = run_dir / "ai_packs" / "note_style" / "packs"
    run_dir.mkdir(parents=True)
    _write_pack(packs_dir / "acc_idx-001.jsonl")
    _write_runflow(run_dir / "runflow.json", "success")

    updated = backfill_note_style_runflow(tmp_path)

    assert updated == [sid]

    payload = json.loads((run_dir / "runflow.json").read_text(encoding="utf-8"))
    stage = payload["stages"]["note_style"]
    assert stage["status"] == "built"
    assert stage["sent"] is False
    assert stage["completed_at"] is None
    assert stage["results"]["completed"] == 0
    assert stage["results"]["failed"] == 0
    assert stage["summary"]["completed"] == 0
    assert stage["summary"]["failed"] == 0


def test_backfill_dry_run_does_not_write(tmp_path: Path) -> None:
    sid = "SID002"
    run_dir = tmp_path / sid
    packs_dir = run_dir / "ai_packs" / "note_style" / "packs"
    run_dir.mkdir(parents=True)
    _write_pack(packs_dir / "acc_idx-002.jsonl")
    _write_runflow(run_dir / "runflow.json", "success")

    original = (run_dir / "runflow.json").read_text(encoding="utf-8")

    updated = backfill_note_style_runflow(tmp_path, dry_run=True)

    assert updated == [sid]
    assert (run_dir / "runflow.json").read_text(encoding="utf-8") == original
