from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from backend.ai.note_style.io import note_style_stage_view
from backend.core.ai.paths import (
    ensure_note_style_paths,
    note_style_pack_filename,
    note_style_result_filename,
)


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")


def prime_stage(
    runs_root: Path,
    sid: str,
    *,
    expected_accounts: Sequence[str] | None = None,
    built_accounts: Iterable[str] | None = None,
    completed_accounts: Iterable[str] | None = None,
    failed_accounts: Iterable[str] | None = None,
) -> None:
    """Create pack and result artifacts for ``sid`` under ``runs_root``."""

    expected = list(expected_accounts or [])
    built = set(built_accounts or [])
    completed = set(completed_accounts or [])
    failed = set(failed_accounts or [])

    run_dir = runs_root / sid
    responses_dir = run_dir / "frontend" / "review" / "responses"

    for account_id in expected:
        _write_json(
            responses_dir / f"{account_id}.result.json",
            {"note": f"Customer note for {account_id}"},
        )

    paths = ensure_note_style_paths(runs_root, sid, create=True)

    for account_id in built:
        pack_payload = {
            "account_id": account_id,
            "note_text": f"Customer note for {account_id}",
            "messages": [
                {"role": "system", "content": "test"},
                {"role": "user", "content": f"payload:{account_id}"},
            ],
        }
        pack_path = paths.packs_dir / note_style_pack_filename(account_id)
        _write_json(pack_path, pack_payload)

    for account_id in completed:
        result_path = paths.results_dir / note_style_result_filename(account_id)
        _write_json(result_path, {"status": "completed", "analysis": {"note": "ok"}})

    for account_id in failed:
        result_path = paths.results_dir / note_style_result_filename(account_id)
        _write_json(result_path, {"status": "failed", "error": "simulated"})


def stage_view(runs_root: Path, sid: str):
    """Return the lifecycle view for ``sid``."""

    return note_style_stage_view(sid, runs_root=runs_root)

