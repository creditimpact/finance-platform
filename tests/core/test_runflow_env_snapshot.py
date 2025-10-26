from __future__ import annotations

from pathlib import Path

from backend.core.runflow.env_snapshot import collect_process_snapshot


def test_collect_process_snapshot_validation_paths(monkeypatch, tmp_path):
    runs_root = tmp_path / "runs"
    runs_root.mkdir()

    sid = "2024-001"
    monkeypatch.setenv("RUNS_ROOT", str(runs_root))
    monkeypatch.setenv("ENABLE_FRONTEND_PACKS", "0")
    monkeypatch.setenv("ENABLE_VALIDATION_SENDER", "1")
    monkeypatch.setenv("AUTO_VALIDATION_SEND", "yes")
    monkeypatch.setenv("VALIDATION_SEND_ON_BUILD", "no")
    monkeypatch.setenv("VALIDATION_AUTOSEND_ENABLED", "no")
    monkeypatch.setenv("VALIDATION_PACKS_DIR", "custom/packs")
    monkeypatch.setenv("VALIDATION_API_KEY", "top-secret")

    snapshot = collect_process_snapshot(stage="validation", sid=sid)

    flags = snapshot["flags"]
    assert flags["ENABLE_FRONTEND_PACKS"]["enabled"] is False
    assert flags["ENABLE_VALIDATION_SENDER"]["enabled"] is True
    assert flags["AUTO_VALIDATION_SEND"]["enabled"] is True
    assert flags["VALIDATION_SEND_ON_BUILD"]["enabled"] is False
    assert flags["VALIDATION_AUTOSEND_ENABLED"]["enabled"] is False

    validation_paths = snapshot["paths"]["validation"]
    packs_dir = Path(validation_paths["packs_dir"])
    assert "custom" in packs_dir.parts
    assert packs_dir.is_absolute()

    validation_env = snapshot["validation_env"]
    assert validation_env["VALIDATION_PACKS_DIR"] == "custom/packs"
    assert validation_env["VALIDATION_API_KEY"] == "<redacted>"


def test_collect_process_snapshot_templates(monkeypatch, tmp_path):
    runs_root = tmp_path / "runs"
    runs_root.mkdir()

    monkeypatch.setenv("RUNS_ROOT", str(runs_root))

    snapshot = collect_process_snapshot()

    templates = snapshot["paths"]["templates"]
    assert templates["frontend_dir"].endswith("<sid>/frontend")
    assert templates["merge_packs"].endswith("<sid>/ai_packs/merge/packs")
    assert templates["validation_results"].endswith(
        "<sid>/ai_packs/validation/results"
    )
