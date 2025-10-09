import json
from datetime import datetime
from pathlib import Path

import pytest

from backend.validation.tolerance import clear_cached_conventions
from backend.validation.utils_amounts import are_amounts_within_tolerance
from backend.validation.utils_dates import (
    are_dates_within_tolerance,
    load_date_convention_for_sid,
    parse_date_with_convention,
)


@pytest.fixture(autouse=True)
def _clear_cache():
    """Ensure cached date conventions are cleared between tests."""

    clear_cached_conventions()
    yield
    clear_cached_conventions()


class TestDateParsing:
    def test_iso_parsing_ignores_convention(self):
        iso_value = "2024-03-05"
        parsed_mdy = parse_date_with_convention(iso_value, "MDY")
        parsed_dmy = parse_date_with_convention(iso_value, "DMY")

        assert parsed_mdy == datetime(2024, 3, 5)
        assert parsed_dmy == datetime(2024, 3, 5)

    def test_mdy_and_dmy_disambiguation(self):
        value = "03/04/2024"

        parsed_mdy = parse_date_with_convention(value, "MDY")
        parsed_dmy = parse_date_with_convention(value, "DMY")

        assert parsed_mdy == datetime(2024, 3, 4)
        assert parsed_dmy == datetime(2024, 4, 3)


class TestDateTolerance:
    def test_spans_within_and_beyond_tolerance(self):
        within_values = ["2024-05-01", "2024-05-05", "2024-05-04"]
        beyond_values = ["2024-05-01", "2024-05-10"]

        within_result, within_span = are_dates_within_tolerance(within_values, "MDY", 5)
        beyond_result, beyond_span = are_dates_within_tolerance(beyond_values, "MDY", 5)

        assert within_result is True
        assert within_span == 4

        assert beyond_result is False
        assert beyond_span == 9

    def test_zero_or_one_valid_date_is_within_tolerance(self):
        values = [None, "", "not-a-date"]

        within, span = are_dates_within_tolerance(values, "MDY", 5)

        assert within is True
        assert span is None


class TestAmountTolerance:
    def test_within_absolute_tolerance(self):
        within, diff, max_value = are_amounts_within_tolerance([100, 118], 25.0, 0.0)

        assert within is True
        assert diff == pytest.approx(18.0)
        assert max_value == pytest.approx(118.0)

    def test_within_ratio_tolerance(self):
        within, diff, max_value = are_amounts_within_tolerance([100, 110], 5.0, 0.2)

        assert within is True
        assert diff == pytest.approx(10.0)
        assert max_value == pytest.approx(110.0)

    def test_beyond_all_tolerance_thresholds(self):
        within, diff, max_value = are_amounts_within_tolerance([100, 200], 20.0, 0.1)

        assert within is False
        assert diff == pytest.approx(100.0)
        assert max_value == pytest.approx(200.0)

    def test_zero_or_one_numeric_value_is_within_tolerance(self):
        within, diff, max_value = are_amounts_within_tolerance([None, "not-a-number"], 50.0, 0.01)

        assert within is True
        assert diff is None
        assert max_value is None


class TestLoadDateConvention:
    def _write_convention_file(self, runs_root: Path, sid: str, payload: dict[str, str]) -> Path:
        sid_root = runs_root / sid / "trace"
        sid_root.mkdir(parents=True, exist_ok=True)
        file_path = sid_root / "date_convention.json"
        file_path.write_text(json.dumps(payload), encoding="utf-8")
        return file_path

    def test_loads_dmy_convention_from_trace(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("PREVALIDATION_OUT_PATH_REL", raising=False)
        runs_root = tmp_path
        sid = "SID123"
        self._write_convention_file(runs_root, sid, {"convention": "DMY"})

        convention = load_date_convention_for_sid(str(runs_root), sid)

        assert convention == "DMY"

    def test_loads_mdy_convention_from_trace_short_key(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("PREVALIDATION_OUT_PATH_REL", raising=False)
        runs_root = tmp_path
        sid = "SID999"
        self._write_convention_file(runs_root, sid, {"conv": "MDY"})

        convention = load_date_convention_for_sid(str(runs_root), sid)

        assert convention == "MDY"
