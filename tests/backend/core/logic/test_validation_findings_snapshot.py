import json
import shutil
from pathlib import Path

import pytest

from backend.core.logic.validation_requirements import (
    build_validation_requirements_for_account,
)


_FIXTURE_ROOT = Path(
    "runs/a09686e7-11b9-47a8-a5a0-0fdabc20e220/cases/accounts"
)
_SNAPSHOT_ROOT = Path(__file__).parent / "snapshots" / "validation_findings"


def _normalize_findings(findings):
    normalized = json.loads(json.dumps(findings, sort_keys=True))
    normalized.sort(key=lambda entry: entry.get("field", ""))
    return normalized


@pytest.mark.parametrize(
    "account_label, account_id",
    [
        ("account-8", "8"),
        ("account-11", "11"),
        ("account-12", "12"),
        ("account-16", "16"),
    ],
)
def test_validation_findings_match_snapshot(account_label, account_id, tmp_path):
    fixture_dir = _FIXTURE_ROOT / account_id
    if not fixture_dir.exists():
        pytest.skip(f"fixture directory missing: {fixture_dir}")

    working_dir = tmp_path / account_id
    shutil.copytree(fixture_dir, working_dir)

    result = build_validation_requirements_for_account(working_dir, build_pack=False)
    findings = result["validation_requirements"]["findings"]

    snapshot = {
        "findings": _normalize_findings(findings),
        "ai_fields": sorted(
            entry["field"] for entry in findings if entry.get("send_to_ai")
        ),
    }

    snapshot_path = _SNAPSHOT_ROOT / f"{account_label}.json"
    expected = json.loads(snapshot_path.read_text(encoding="utf-8"))

    assert snapshot == expected
