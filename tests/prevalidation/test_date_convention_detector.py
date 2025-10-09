import json
from pathlib import Path

import pytest

from backend.prevalidation.date_convention_detector import detect_month_language_for_run
from backend.prevalidation.tasks import detect_and_persist_date_convention


@pytest.fixture
def run_dir(tmp_path: Path) -> Path:
    run_directory = tmp_path / "run"
    (run_directory / "cases" / "accounts").mkdir(parents=True, exist_ok=True)
    return run_directory


def _write_raw_lines(account_dir: Path, texts: list[str]) -> None:
    account_dir.mkdir(parents=True, exist_ok=True)
    raw_lines_path = account_dir / "raw_lines.json"

    parts: list[str] = ["[\n"]
    for idx, text in enumerate(texts):
        serialized = json.dumps(text, ensure_ascii=False)
        parts.append("  {\n    \"text\": " + serialized + "\n  }")
        if idx < len(texts) - 1:
            parts.append(",\n")
        else:
            parts.append("\n")
    parts.append("]\n")

    raw_lines_path.write_text("".join(parts), encoding="utf-8")


def test_detect_month_language_hebrew_accounts(run_dir: Path) -> None:
    accounts_dir = run_dir / "cases" / "accounts"

    _write_raw_lines(
        accounts_dir / "account_1",
        [
            "Intro text",
            "Two-Year Payment History  ספט׳ דצמ׳",
            "Supplemental text",
        ],
    )

    _write_raw_lines(
        accounts_dir / "account_2",
        [
            "Two-Year Payment History  אפר׳ יוני",
        ],
    )

    result = detect_month_language_for_run(str(run_dir))
    block = result["date_convention"]

    assert block["month_language"] == "he"
    assert block["convention"] == "DMY"
    assert block["confidence"] == 1.0
    assert block["evidence_counts"] == {
        "he_hits": 4,
        "en_hits": 0,
        "accounts_scanned": 2,
    }


def test_detect_month_language_english_only(run_dir: Path) -> None:
    accounts_dir = run_dir / "cases" / "accounts"

    _write_raw_lines(
        accounts_dir / "account_1",
        [
            "Two-Year Payment History Jan Feb Mar",
        ],
    )

    _write_raw_lines(
        accounts_dir / "account_2",
        [
            "Other header",
            "Two-Year Payment History April May",
        ],
    )

    result = detect_month_language_for_run(str(run_dir))
    block = result["date_convention"]

    assert block["month_language"] == "en"
    assert block["convention"] == "MDY"
    assert block["confidence"] == 1.0
    assert block["evidence_counts"] == {
        "he_hits": 0,
        "en_hits": 5,
        "accounts_scanned": 2,
    }


def test_detect_month_language_skips_accounts_without_marker(run_dir: Path) -> None:
    accounts_dir = run_dir / "cases" / "accounts"

    _write_raw_lines(
        accounts_dir / "account_with_marker",
        [
            "Two-Year Payment History October November",
        ],
    )

    _write_raw_lines(
        accounts_dir / "account_without_marker",
        [
            "October November December",  # Should be ignored because marker is missing.
        ],
    )

    result = detect_month_language_for_run(str(run_dir))
    block = result["date_convention"]

    assert block["month_language"] == "en"
    assert block["convention"] == "MDY"
    assert block["confidence"] == 1.0
    assert block["evidence_counts"] == {
        "he_hits": 0,
        "en_hits": 2,
        "accounts_scanned": 2,
    }


def test_detect_month_language_tie_or_no_hits(run_dir: Path) -> None:
    accounts_dir = run_dir / "cases" / "accounts"

    _write_raw_lines(
        accounts_dir / "account_1",
        [
            "Two-Year Payment History",  # Marker without months.
        ],
    )

    result = detect_month_language_for_run(str(run_dir))
    block = result["date_convention"]

    assert block["month_language"] == "unknown"
    assert block["convention"] is None
    assert block["confidence"] == 0.0
    assert block["evidence_counts"] == {
        "he_hits": 0,
        "en_hits": 0,
        "accounts_scanned": 1,
    }


def test_detect_and_persist_writes_date_convention_file(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    sid = "SID123"
    run_dir = runs_root / sid
    accounts_dir = run_dir / "cases" / "accounts"
    _write_raw_lines(accounts_dir / "account_1", ["Two-Year Payment History Jan Feb Mar"])

    block = detect_and_persist_date_convention(sid, runs_root=runs_root)
    assert block is not None

    output_path = run_dir / "traces" / "date_convention.json"
    assert output_path.exists()

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload == {
        "date_convention": {
            "scope": "global",
            "convention": "MDY",
            "month_language": "en",
            "confidence": 1.0,
            "evidence_counts": {
                "he_hits": 0,
                "en_hits": 3,
                "accounts_scanned": 1,
            },
            "detector_version": "1.0",
        }
    }

    legacy_path = run_dir / "traces" / "accounts_table" / "general_info_from_full.json"
    assert not legacy_path.exists()


def test_detect_and_persist_is_idempotent(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    sid = "SID456"
    run_dir = runs_root / sid
    accounts_dir = run_dir / "cases" / "accounts"
    _write_raw_lines(accounts_dir / "account_1", ["Two-Year Payment History Jan Feb Mar"])

    first_block = detect_and_persist_date_convention(sid, runs_root=runs_root)
    assert first_block is not None

    output_path = run_dir / "traces" / "date_convention.json"
    first_mtime = output_path.stat().st_mtime_ns

    second_block = detect_and_persist_date_convention(sid, runs_root=runs_root)
    assert second_block == first_block
    second_mtime = output_path.stat().st_mtime_ns

    assert second_mtime == first_mtime


def test_detect_and_persist_preserves_general_info_mtime(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    sid = "SID789"
    run_dir = runs_root / sid
    accounts_dir = run_dir / "cases" / "accounts"
    _write_raw_lines(accounts_dir / "account_1", ["Two-Year Payment History Jan Feb Mar"])

    legacy_dir = run_dir / "traces" / "accounts_table"
    legacy_dir.mkdir(parents=True, exist_ok=True)
    legacy_path = legacy_dir / "general_info_from_full.json"
    legacy_path.write_text("{\n  \"foo\": \"bar\"\n}\n", encoding="utf-8")

    before_mtime = legacy_path.stat().st_mtime_ns

    detect_and_persist_date_convention(sid, runs_root=runs_root)
    after_first_run = legacy_path.stat().st_mtime_ns
    assert after_first_run == before_mtime

    detect_and_persist_date_convention(sid, runs_root=runs_root)
    after_second_run = legacy_path.stat().st_mtime_ns
    assert after_second_run == before_mtime
