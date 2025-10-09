import json
from pathlib import Path
from typing import Any

import pytest

from backend.validation.utils_dates import load_date_convention_for_sid


@pytest.fixture
def sample_run_dir(tmp_path: Path) -> Path:
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    return runs_root


def write_convention_file(runs_root: Path, sid: str, rel_path: str, payload: dict) -> Path:
    convention_path = runs_root / sid / rel_path
    convention_path.parent.mkdir(parents=True, exist_ok=True)
    convention_path.write_text(json.dumps(payload), encoding="utf-8")
    return convention_path


def write_manifest(
    runs_root: Path,
    sid: str,
    *,
    traces_rel: str | None = None,
    prevalidation_block: dict | None = None,
) -> Path:
    manifest_path = runs_root / sid / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {"sid": sid}
    if traces_rel is not None:
        payload.setdefault("artifacts", {}).setdefault("traces", {})[
            "date_convention_rel"
        ] = traces_rel
    if prevalidation_block is not None:
        payload.setdefault("prevalidation", {})["date_convention"] = dict(
            prevalidation_block
        )
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    return manifest_path


def test_load_date_convention_from_trace_file(sample_run_dir: Path) -> None:
    sid = "SID123"
    rel_path = "trace/date_convention.json"
    write_convention_file(sample_run_dir, sid, rel_path, {"convention": "DMY"})

    convention = load_date_convention_for_sid(str(sample_run_dir), sid)

    assert convention == "DMY"


def test_load_date_convention_missing_file_defaults(sample_run_dir: Path) -> None:
    sid = "SID124"
    write_manifest(
        sample_run_dir,
        sid,
        prevalidation_block={"convention": "DMY", "file_rel": "trace/date_convention.json"},
    )

    convention = load_date_convention_for_sid(str(sample_run_dir), sid)

    assert convention == "MDY"


def test_load_date_convention_with_custom_rel_path(sample_run_dir: Path) -> None:
    sid = "SID125"
    rel_path = "custom/output.json"
    write_convention_file(sample_run_dir, sid, rel_path, {"conv": "ymd"})

    convention = load_date_convention_for_sid(str(sample_run_dir), sid, rel_path=rel_path)

    assert convention == "YMD"


def test_load_date_convention_invalid_payload_defaults(sample_run_dir: Path) -> None:
    sid = "SID126"
    rel_path = "trace/date_convention.json"
    write_convention_file(sample_run_dir, sid, rel_path, {"convention": "unknown"})

    convention = load_date_convention_for_sid(str(sample_run_dir), sid)

    assert convention == "MDY"


def test_load_date_convention_uses_manifest_relative_path(sample_run_dir: Path) -> None:
    sid = "SID127"
    manifest_rel = "alt/location/date.json"
    write_manifest(
        sample_run_dir,
        sid,
        traces_rel=manifest_rel,
        prevalidation_block={"convention": "DMY"},
    )
    write_convention_file(sample_run_dir, sid, manifest_rel, {"convention": "YMD"})

    convention = load_date_convention_for_sid(str(sample_run_dir), sid)

    assert convention == "YMD"


def test_load_date_convention_uses_prevalidation_file_rel(sample_run_dir: Path) -> None:
    sid = "SID128"
    manifest_rel = "trace\\date_convention.json"
    write_manifest(
        sample_run_dir,
        sid,
        prevalidation_block={"file_rel": manifest_rel},
    )
    write_convention_file(sample_run_dir, sid, "trace/date_convention.json", {"conv": "dmy"})

    convention = load_date_convention_for_sid(str(sample_run_dir), sid)

    assert convention == "DMY"
