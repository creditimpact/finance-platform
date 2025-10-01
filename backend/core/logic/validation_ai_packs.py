"""Helpers for building validation AI adjudication packs."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Sequence

from backend.core.ai.paths import (
    ValidationAccountPaths,
    ensure_validation_account_paths,
    ensure_validation_paths,
)
from backend.pipeline.runs import RunManifest

log = logging.getLogger(__name__)


def _normalize_indices(indices: Iterable[int | str]) -> list[int]:
    normalized: set[int] = set()
    for idx in indices:
        try:
            normalized.add(int(str(idx)))
        except Exception:
            continue
    return sorted(normalized)


def build_validation_ai_packs_for_accounts(
    sid: str,
    *,
    account_indices: Sequence[int | str],
    runs_root: Path | str | None = None,
) -> None:
    """Trigger validation AI pack building for the provided account indices.

    The builder currently ensures the filesystem scaffold for validation AI
    packs exists so subsequent stages can populate payloads and prompts.
    """

    normalized_indices = _normalize_indices(account_indices)
    if not normalized_indices:
        return

    runs_root_path = Path(runs_root) if runs_root is not None else Path("runs")
    validation_paths = ensure_validation_paths(runs_root_path, sid, create=True)

    created: list[ValidationAccountPaths] = []
    for idx in normalized_indices:
        account_paths = ensure_validation_account_paths(
            validation_paths, idx, create=True
        )
        _ensure_placeholder_files(account_paths)
        created.append(account_paths)

    manifest_path = runs_root_path / sid / "manifest.json"
    manifest = RunManifest.load_or_create(manifest_path, sid)
    if created:
        last_account = created[-1]
        index_path = validation_paths.base / "index.json"
        manifest.upsert_validation_packs_dir(
            validation_paths.base,
            account_dir=last_account.base,
            results_dir=last_account.results_dir,
            index_file=index_path,
        )
    else:
        manifest.upsert_validation_packs_dir(validation_paths.base)

    log.info(
        "VALIDATION_AI_PACKS_INITIALIZED sid=%s base=%s accounts=%s",
        sid,
        validation_paths.base,
        ",".join(str(path.base.name) for path in created),
    )


def _ensure_placeholder_files(paths: ValidationAccountPaths) -> None:
    """Create empty scaffold files for a validation pack if they are missing."""

    _ensure_file(paths.pack_file, "{}\n")
    _ensure_file(paths.prompt_file, "")
    _ensure_file(paths.model_results_file, "{}\n")


def _ensure_file(path: Path, default_contents: str) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(default_contents, encoding="utf-8")

