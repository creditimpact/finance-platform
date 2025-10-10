from __future__ import annotations

import json
from pathlib import Path

from backend.frontend.packs.generator import generate_frontend_packs_for_run


def _write_summary(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_generate_frontend_packs_writes_payload(tmp_path):
    runs_root = tmp_path / "runs"
    sid = "S100"
    summary_path = (
        runs_root
        / sid
        / "cases"
        / "accounts"
        / "1"
        / "summary.json"
    )
    summary_payload = {
        "account_id": "acct-1",
        "account_index": 1,
        "validation_requirements": {
            "findings": [
                {"id": "f1", "details": {"reason": "mismatch", "raw": {"keep": "x"}}},
                {"id": "f2", "raw": {"drop": True}},
            ]
        },
    }
    _write_summary(summary_path, summary_payload)

    result = generate_frontend_packs_for_run(sid, runs_root=runs_root)

    output_path = runs_root / sid / "frontend" / "packs.json"
    assert output_path.exists()

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["sid"] == sid
    assert payload["packs_count"] == 2
    assert payload["accounts_with_findings"] == 1
    assert payload["accounts"][0]["account_id"] == "acct-1"
    assert "raw" not in json.dumps(payload["accounts"][0])

    assert result["packs_count"] == 2
    assert Path(result["path"]).resolve() == output_path.resolve()
