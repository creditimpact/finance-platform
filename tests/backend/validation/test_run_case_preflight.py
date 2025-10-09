from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

import pytest

run_case_module = importlib.import_module("backend.validation.run_case")


def _write_manifest(tmp_path: Path, sid: str) -> Path:
    run_dir = tmp_path / sid
    accounts_dir = run_dir / "cases" / "accounts"
    packs_dir = run_dir / "ai_packs" / "validation" / "packs"
    results_dir = run_dir / "ai_packs" / "validation" / "results"
    index_path = run_dir / "ai_packs" / "validation" / "index.json"
    log_path = run_dir / "ai_packs" / "validation" / "validation.log"

    manifest = {
        "sid": sid,
        "base_dirs": {"cases_accounts_dir": str(accounts_dir)},
        "ai": {
            "packs": {
                "validation": {
                    "packs_dir": str(packs_dir),
                    "results_dir": str(results_dir),
                    "index": str(index_path),
                    "logs": str(log_path),
                }
            }
        },
    }

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    accounts_dir.mkdir(parents=True, exist_ok=True)
    return manifest_path


def _patch_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    def _summary_stub(_manifest: Any, *, cfg: Any) -> dict[str, Any]:
        return {
            "total_accounts": 0,
            "summaries_written": 0,
            "packs_built": 0,
            "skipped_accounts": 0,
            "errors": 0,
        }

    monkeypatch.setattr(
        "backend.validation.pipeline.run_validation_summary_pipeline",
        _summary_stub,
    )
    monkeypatch.setattr(
        run_case_module,
        "send_validation_packs",
        lambda _manifest: [],
    )


def test_run_case_invokes_detection_when_enabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path = _write_manifest(tmp_path, "S123")
    _patch_pipeline(monkeypatch)

    calls: list[tuple[str, Path | None]] = []

    def _detect(sid: str, *, runs_root: Path | str | None = None):  # type: ignore[override]
        calls.append((sid, Path(runs_root) if runs_root is not None else None))
        return {"convention": "MDY"}

    monkeypatch.setattr(run_case_module, "detect_dates_for_sid", _detect)
    monkeypatch.setattr(run_case_module, "PREVALIDATION_DETECT_DATES", True)
    monkeypatch.setattr(run_case_module, "ENABLE_VALIDATION_AI", True)

    result = run_case_module.run_case(manifest_path)
    assert result["enabled"] is True
    assert calls == [("S123", tmp_path)]


def test_run_case_skips_detection_when_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path = _write_manifest(tmp_path, "S999")
    _patch_pipeline(monkeypatch)

    def _detect(*_args: Any, **_kwargs: Any):  # pragma: no cover - defensive guard
        raise AssertionError("Detection should be skipped")

    monkeypatch.setattr(run_case_module, "detect_dates_for_sid", _detect)
    monkeypatch.setattr(run_case_module, "PREVALIDATION_DETECT_DATES", False)
    monkeypatch.setattr(run_case_module, "ENABLE_VALIDATION_AI", True)

    result = run_case_module.run_case(manifest_path)
    assert result["enabled"] is True
