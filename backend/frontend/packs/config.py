"""Environment-driven configuration for frontend review pack generation."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class FrontendStageConfig:
    """Resolved filesystem locations for the frontend review stage."""

    stage_dir: Path
    packs_dir: Path
    responses_dir: Path
    index_path: Path


def _resolve_path(run_dir: Path, env_name: str, *, default: str) -> Path:
    value = os.getenv(env_name)
    candidate = Path(value) if value else Path(default)
    if candidate.is_absolute():
        return candidate
    return run_dir / candidate


def load_frontend_stage_config(run_dir: Path | str) -> FrontendStageConfig:
    """Load the configured frontend review stage paths."""

    base_dir = Path(run_dir)

    stage_dir = _resolve_path(base_dir, "FRONTEND_PACKS_STAGE_DIR", default="frontend/review")
    packs_dir = _resolve_path(base_dir, "FRONTEND_PACKS_DIR", default="frontend/review/packs")
    responses_dir = _resolve_path(
        base_dir, "FRONTEND_PACKS_RESPONSES_DIR", default="frontend/review/responses"
    )
    index_path = _resolve_path(
        base_dir, "FRONTEND_PACKS_INDEX", default="frontend/review/index.json"
    )

    return FrontendStageConfig(
        stage_dir=stage_dir,
        packs_dir=packs_dir,
        responses_dir=responses_dir,
        index_path=index_path,
    )

