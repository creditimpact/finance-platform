from __future__ import annotations

import json
from pathlib import Path

from backend.runflow.manifest import (
    update_manifest_frontend,
    update_manifest_state,
)


def _load_manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_update_manifest_state_sets_status_and_run_state(tmp_path):
    runs_root = tmp_path / "runs"
    sid = "S1000"

    manifest = update_manifest_state(
        sid,
        "AWAITING_CUSTOMER_INPUT",
        runs_root=runs_root,
    )

    manifest_path = runs_root / sid / "manifest.json"
    assert manifest_path.exists()
    payload = _load_manifest(manifest_path)

    assert payload["status"] == "AWAITING_CUSTOMER_INPUT"
    assert payload["run_state"] == "AWAITING_CUSTOMER_INPUT"
    assert manifest.data["run_state"] == "AWAITING_CUSTOMER_INPUT"


def test_update_manifest_frontend_persists_section(tmp_path):
    runs_root = tmp_path / "runs"
    sid = "S2000"

    manifest = update_manifest_state(
        sid,
        "VALIDATING",
        runs_root=runs_root,
    )

    packs_dir = runs_root / sid / "frontend"
    result = update_manifest_frontend(
        sid,
        packs_dir=packs_dir,
        packs_count=3,
        built=True,
        last_built_at="2024-01-01T00:00:00Z",
        manifest=manifest,
    )

    manifest_path = runs_root / sid / "manifest.json"
    payload = _load_manifest(manifest_path)
    frontend_section = payload.get("frontend")

    assert isinstance(frontend_section, dict)
    assert frontend_section["built"] is True
    assert frontend_section["packs_count"] == 3
    assert frontend_section["counts"] == {"packs": 3, "responses": 0}
    assert frontend_section["last_built_at"] == "2024-01-01T00:00:00Z"
    assert frontend_section["last_responses_at"]

    assert frontend_section["base"].endswith("frontend")
    assert frontend_section["dir"].endswith("frontend/review")
    assert frontend_section["packs"].endswith("frontend/review/packs")
    assert frontend_section["packs_dir"].endswith("frontend/review/packs")
    assert frontend_section["results"].endswith("frontend/review/responses")
    assert frontend_section["results_dir"].endswith("frontend/review/responses")
    assert frontend_section["index"].endswith("frontend/review/index.json")
    assert frontend_section["legacy_index"].endswith("frontend/index.json")

    assert result.data["frontend"] == frontend_section
