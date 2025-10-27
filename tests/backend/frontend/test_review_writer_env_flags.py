"""Tests for ``backend.frontend.review_writer`` environment flag gating."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from backend.frontend import review_writer


@pytest.fixture(name="review_index_path")
def _review_index_path(tmp_path: Path) -> Path:
    sid = "SID-456"
    index_dir = tmp_path / sid / "frontend" / "review"
    index_dir.mkdir(parents=True, exist_ok=True)
    index_path = index_dir / "index.json"
    index_path.write_text("{""records"": []}", encoding="utf-8")
    return index_path


def test_maybe_trigger_note_style_respects_env(monkeypatch, review_index_path: Path) -> None:
    """The NOTE_STYLE_SEND_ON_RESPONSE_WRITE flag should gate task scheduling."""

    captured: list[Any] = []

    class _TaskStub:
        def delay(self, sid: str) -> None:
            captured.append(sid)

    monkeypatch.setattr(review_writer, "note_style_prepare_and_send_task", _TaskStub())
    monkeypatch.setenv("RUNS_ROOT", str(review_index_path.parents[3]))
    monkeypatch.setenv("NOTE_STYLE_INDEX_MIN_BYTES", "1")

    monkeypatch.setenv("NOTE_STYLE_SEND_ON_RESPONSE_WRITE", "0")
    review_writer.maybe_trigger_note_style_on_response_write("SID-456")
    assert captured == []

    monkeypatch.setenv("NOTE_STYLE_SEND_ON_RESPONSE_WRITE", "1")
    review_writer.maybe_trigger_note_style_on_response_write("SID-456")
    assert captured == ["SID-456"]
