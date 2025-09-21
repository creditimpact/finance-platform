"""Automatic AI adjudication hooks for the case-build pipeline."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from backend.core.logic.report_analysis.tags_compact import (
    compact_tags_for_account,
)
from backend.pipeline.runs import RunManifest
from scripts.build_ai_merge_packs import main as build_ai_merge_packs_main
from scripts.score_bureau_pairs import score_accounts
from scripts.send_ai_merge_packs import main as send_ai_merge_packs_main

logger = logging.getLogger(__name__)


def maybe_run_auto_ai_pipeline(
    sid: str, *, summary: Mapping[str, object] | None = None
) -> None:
    """Run the automatic AI pipeline when enabled via the environment flag."""

    if os.getenv("ENABLE_AUTO_AI_PIPELINE", "0") != "1":
        return

    _run_auto_ai_pipeline(sid, summary=summary)


def _run_auto_ai_pipeline(
    sid: str, *, summary: Mapping[str, object] | None = None
) -> None:
    manifest = RunManifest.for_sid(sid)
    run_dir = manifest.path.parent
    runs_root = run_dir.parent

    accounts_dir = _resolve_accounts_dir(run_dir, summary)
    touched_accounts: set[int] = set()
    index_entries: list[Mapping[str, object]] = []

    logger.info("AUTO_AI_PIPELINE start sid=%s", sid)

    try:
        scoring = score_accounts(sid, runs_root=runs_root, write_tags=True)
        touched_accounts.update(_normalize_indices(scoring.indices))

        _build_ai_packs(sid, runs_root)

        index_path = run_dir / "ai_packs" / "index.json"
        index_entries = _load_ai_index(index_path)
        touched_accounts.update(_indices_from_index(index_entries))

        _send_ai_packs(sid)

        _compact_accounts(accounts_dir, touched_accounts)
    except Exception:
        logger.error("AUTO_AI_PIPELINE failed sid=%s", sid, exc_info=True)
        raise
    else:
        logger.info(
            "AUTO_AI_PIPELINE done sid=%s accounts=%d packs=%d",
            sid,
            len(touched_accounts),
            len(index_entries),
        )


def _build_ai_packs(sid: str, runs_root: Path) -> None:
    argv = ["--sid", sid, "--runs-root", str(runs_root)]
    try:
        build_ai_merge_packs_main(argv)
    except SystemExit as exc:  # pragma: no cover - defensive
        if exc.code not in (None, 0):
            raise RuntimeError(
                f"build_ai_merge_packs failed for sid={sid} exit={exc.code}"
            ) from exc


def _send_ai_packs(sid: str) -> None:
    argv = ["--sid", sid]
    try:
        send_ai_merge_packs_main(argv)
    except SystemExit as exc:
        if exc.code not in (None, 0):
            raise RuntimeError(
                f"send_ai_merge_packs failed for sid={sid} exit={exc.code}"
            ) from exc


def _normalize_indices(indices: Sequence[object]) -> set[int]:
    normalized: set[int] = set()
    for value in indices:
        try:
            normalized.add(int(value))
        except (TypeError, ValueError):
            continue
    return normalized


def _resolve_accounts_dir(
    run_dir: Path, summary: Mapping[str, object] | None
) -> Path:
    if isinstance(summary, Mapping):
        cases = summary.get("cases")
        if isinstance(cases, Mapping):
            dir_value = cases.get("dir")
            if isinstance(dir_value, (str, Path)) and str(dir_value):
                candidate = Path(dir_value)
                return candidate
    return run_dir / "cases" / "accounts"


def _load_ai_index(path: Path) -> list[Mapping[str, object]]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive
        raise ValueError(f"Invalid AI pack index JSON: {path}") from exc
    if not isinstance(data, list):
        raise ValueError(f"AI pack index must be a list: {path}")
    entries: list[Mapping[str, object]] = []
    for entry in data:
        if isinstance(entry, Mapping):
            entries.append(entry)
    return entries


def _indices_from_index(index_entries: Iterable[Mapping[str, object]]) -> set[int]:
    values: set[int] = set()
    for entry in index_entries:
        for key in ("a", "b"):
            if key not in entry:
                continue
            try:
                values.add(int(entry[key]))
            except (TypeError, ValueError):
                continue
    return values


def _compact_accounts(accounts_dir: Path, indices: Iterable[int]) -> None:
    unique_indices = sorted({int(idx) for idx in indices})
    if not unique_indices:
        return

    for idx in unique_indices:
        account_dir = accounts_dir / f"{idx}"
        if not account_dir.exists():
            continue
        try:
            compact_tags_for_account(account_dir)
        except Exception:  # pragma: no cover - defensive logging
            logger.warning(
                "AUTO_AI_PIPELINE compact failed account=%s dir=%s",
                idx,
                account_dir,
                exc_info=True,
            )

