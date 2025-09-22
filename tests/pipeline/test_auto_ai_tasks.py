"""Unit tests covering the auto AI Celery task helpers."""

from __future__ import annotations

from pathlib import Path

from backend.pipeline import auto_ai_tasks


def test_ai_build_packs_step_skips_without_candidates(monkeypatch, tmp_path: Path) -> None:
    sid = "skip"
    runs_root = tmp_path / "runs"

    recorded_has: list[tuple[str, Path]] = []
    monkeypatch.setattr(
        auto_ai_tasks,
        "has_ai_merge_best_pairs",
        lambda sid_value, runs_root_value: recorded_has.append(
            (sid_value, runs_root_value)
        ) or False,
    )
    build_calls: list[tuple[str, Path]] = []
    monkeypatch.setattr(
        auto_ai_tasks,
        "_build_ai_packs",
        lambda sid_value, runs_root_value: build_calls.append(
            (sid_value, runs_root_value)
        ),
    )

    payload = {
        "sid": sid,
        "runs_root": str(runs_root),
        "touched_accounts": [11, 16],
    }

    result = auto_ai_tasks.ai_build_packs_step.run(payload)

    assert result["ai_index"] == []
    assert result["skip_reason"] == "no_candidates"
    # Ensure we computed the guard exactly once and never invoked the build helper.
    assert recorded_has == [(sid, runs_root)]
    assert build_calls == []


def test_ai_send_packs_step_skips_when_no_packs(monkeypatch, tmp_path: Path) -> None:
    sid = "no-send"
    runs_root = tmp_path / "runs"

    send_calls: list[tuple[str, Path]] = []
    monkeypatch.setattr(
        auto_ai_tasks,
        "_send_ai_packs",
        lambda sid_value, runs_root_value=None: send_calls.append(
            (sid_value, runs_root_value)
        ),
    )

    payload = {
        "sid": sid,
        "runs_root": str(runs_root),
        "ai_index": [],
        "skip_reason": "no_candidates",
    }

    result = auto_ai_tasks.ai_send_packs_step.run(payload)

    assert result["skip_reason"] == "no_candidates"
    assert send_calls == []

