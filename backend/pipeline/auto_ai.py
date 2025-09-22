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
from scripts.send_ai_merge_packs import main as send_ai_merge_packs_main

logger = logging.getLogger(__name__)


def maybe_run_auto_ai_pipeline(
    sid: str, *, summary: Mapping[str, object] | None = None
):
    """Backward-compatible shim that queues the auto-AI pipeline."""

    return maybe_queue_auto_ai_pipeline(sid, summary=summary)


def maybe_queue_auto_ai_pipeline(
    sid: str, *, summary: Mapping[str, object] | None = None
):
    """Queue the automatic AI adjudication pipeline when enabled."""

    if os.getenv("ENABLE_AUTO_AI_PIPELINE", "0") != "1":
        logger.info("AUTO_AI_SKIPPED sid=%s reason=disabled", sid)
        return None

    manifest = RunManifest.for_sid(sid)
    run_dir = manifest.path.parent
    runs_root = run_dir.parent

    if not has_ai_merge_best_pairs(sid, runs_root):
        logger.info("AUTO_AI_SKIPPED sid=%s reason=no_ai_candidates", sid)
        return None

    accounts_dir = _resolve_accounts_dir(run_dir, summary)

    logger.info(
        "AUTO_AI_QUEUING sid=%s runs_root=%s accounts_dir=%s",
        sid,
        runs_root,
        accounts_dir,
    )

    try:
        from backend.pipeline import auto_ai_tasks

        result = auto_ai_tasks.enqueue_auto_ai_pipeline(
            sid=sid,
            runs_root=str(runs_root),
            accounts_dir=str(accounts_dir),
        )
    except Exception:  # pragma: no cover - defensive logging
        logger.error("AUTO_AI_QUEUE_FAILED sid=%s", sid, exc_info=True)
        raise

    logger.info("AUTO_AI_QUEUED sid=%s", sid)
    return result


def has_ai_merge_best_pairs(sid: str, runs_root: Path | str) -> bool:
    """Return ``True`` if any account tags require AI merge adjudication."""

    runs_root_path = Path(runs_root)
    accounts_root = runs_root_path / sid / "cases" / "accounts"
    if not accounts_root.exists():
        return False

    for tags_path in sorted(accounts_root.glob("*/tags.json")):
        try:
            raw = tags_path.read_text(encoding="utf-8")
        except OSError:
            continue
        if not raw.strip():
            continue
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            logger.debug("Skipping invalid JSON at %s", tags_path, exc_info=True)
            continue

        for tag in _iter_tag_entries(payload):
            if _is_ai_merge_best_tag(tag):
                return True

    return False


def _build_ai_packs(sid: str, runs_root: Path) -> None:
    argv = ["--sid", sid, "--runs-root", str(runs_root)]
    try:
        build_ai_merge_packs_main(argv)
    except SystemExit as exc:  # pragma: no cover - defensive
        if exc.code not in (None, 0):
            raise RuntimeError(
                f"build_ai_merge_packs failed for sid={sid} exit={exc.code}"
            ) from exc


def _send_ai_packs(sid: str, runs_root: Path | None = None) -> None:
    argv = ["--sid", sid]
    if runs_root is not None:
        argv.extend(["--runs-root", str(runs_root)])
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


def _iter_tag_entries(payload: object) -> Iterable[Mapping[str, object]]:
    if isinstance(payload, list):
        for entry in payload:
            if isinstance(entry, Mapping):
                yield entry
        return

    if isinstance(payload, Mapping):
        tags = payload.get("tags")
        if isinstance(tags, list):
            for entry in tags:
                if isinstance(entry, Mapping):
                    yield entry


def _is_ai_merge_best_tag(tag: Mapping[str, object]) -> bool:
    if not isinstance(tag, Mapping):
        return False

    kind = str(tag.get("kind", "")).strip().lower()
    if kind != "merge_best":
        return False

    decision_raw = tag.get("decision")
    decision = str(decision_raw).strip().lower() if decision_raw is not None else ""
    if decision != "ai":
        return False

    return tag.get("with") is not None

