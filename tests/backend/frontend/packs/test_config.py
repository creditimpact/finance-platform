from __future__ import annotations

from pathlib import Path

from backend.frontend.packs.config import load_frontend_stage_config


def test_load_frontend_stage_config_defaults(tmp_path, monkeypatch):
    for name in (
        "FRONTEND_PACKS_STAGE_DIR",
        "FRONTEND_PACKS_DIR",
        "FRONTEND_PACKS_RESPONSES_DIR",
        "FRONTEND_PACKS_INDEX",
    ):
        monkeypatch.delenv(name, raising=False)

    run_dir = tmp_path / "runs" / "SID123"
    config = load_frontend_stage_config(run_dir)

    assert config.stage_dir == run_dir / "frontend" / "review"
    assert config.packs_dir == run_dir / "frontend" / "review" / "packs"
    assert config.responses_dir == run_dir / "frontend" / "review" / "responses"
    assert config.index_path == run_dir / "frontend" / "review" / "index.json"


def test_load_frontend_stage_config_env_overrides(tmp_path, monkeypatch):
    monkeypatch.setenv("FRONTEND_PACKS_STAGE_DIR", "custom/review")
    monkeypatch.setenv("FRONTEND_PACKS_DIR", "custom/review/packs")
    monkeypatch.setenv("FRONTEND_PACKS_RESPONSES_DIR", "/tmp/frontend/responses")
    monkeypatch.setenv("FRONTEND_PACKS_INDEX", "custom/review/manifest.json")

    run_dir = tmp_path / "runs" / "SID999"
    config = load_frontend_stage_config(run_dir)

    assert config.stage_dir == run_dir / Path("custom/review")
    assert config.packs_dir == run_dir / Path("custom/review/packs")
    assert config.responses_dir == Path("/tmp/frontend/responses")
    assert config.index_path == run_dir / Path("custom/review/manifest.json")
