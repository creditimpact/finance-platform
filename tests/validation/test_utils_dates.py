import json
from pathlib import Path

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


def test_load_date_convention_from_trace_file(sample_run_dir: Path) -> None:
    sid = "SID123"
    rel_path = "trace/date_convention.json"
    write_convention_file(sample_run_dir, sid, rel_path, {"convention": "DMY"})

    convention = load_date_convention_for_sid(str(sample_run_dir), sid)

    assert convention == "DMY"


def test_load_date_convention_missing_file_defaults(sample_run_dir: Path) -> None:
    sid = "SID124"

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
