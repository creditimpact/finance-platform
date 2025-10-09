from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.validation.preflight import detect_dates_for_sid
from backend.validation.preflight import detect_dates as detect_dates_module


class _Recorder:
    def __init__(self) -> None:
        self.calls: list[tuple[str, Path | None]] = []

    def __call__(self, sid: str, runs_root: Path | str | None = None):  # type: ignore[override]
        root_value: Path | None
        if runs_root is None:
            root_value = None
        else:
            root_value = Path(runs_root)
        self.calls.append((sid, root_value))
        return {"convention": "MDY", "month_language": "en"}


def test_detect_dates_for_sid_invokes_task(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    recorder = _Recorder()
    monkeypatch.setattr(
        detect_dates_module,
        "detect_and_persist_date_convention",
        recorder,
    )

    result = detect_dates_for_sid("SID123", runs_root=tmp_path)
    assert result == {"convention": "MDY", "month_language": "en"}
    assert recorder.calls == [("SID123", tmp_path)]


def test_detect_dates_for_sid_requires_sid() -> None:
    with pytest.raises(ValueError):
        detect_dates_for_sid("")


def test_cli_main_runs_detector(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    recorder = _Recorder()
    monkeypatch.setattr(
        detect_dates_module,
        "detect_and_persist_date_convention",
        recorder,
    )

    exit_code = detect_dates_module.main(
        ["--sid", "SID999", "--runs-root", str(tmp_path)]
    )
    assert exit_code == 0

    assert recorder.calls == [("SID999", tmp_path)]

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["sid"] == "SID999"
    assert payload["date_convention"] == {"convention": "MDY", "month_language": "en"}
