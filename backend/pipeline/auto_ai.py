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

PIPELINE_MARKER_FILENAME = "auto_ai_pipeline_in_progress.json"


def _pipeline_marker_path(runs_root: Path, sid: str) -> Path:
    return runs_root / sid / "ai_packs" / PIPELINE_MARKER_FILENAME


def maybe_run_auto_ai_pipeline(
    sid: str, *, summary: Mapping[str, object] | None = None
):
    """Backward-compatible shim that queues the auto-AI pipeline."""

    _ = summary  # preserved for compatibility with older call sites
    manifest = RunManifest.for_sid(sid)
    runs_root = manifest.path.parent.parent
    return maybe_queue_auto_ai_pipeline(
        sid,
        runs_root=runs_root,
        flag_env=os.environ,
    )


def maybe_queue_auto_ai_pipeline(
    sid: str,
    *,
    runs_root: Path,
    flag_env: Mapping[str, str],
) -> dict[str, object]:
    """Queue the automatic AI adjudication pipeline when enabled."""

    flag_value = str(flag_env.get("ENABLE_AUTO_AI_PIPELINE", "0"))
    if flag_value != "1":
        logger.info("AUTO_AI_SKIP_DISABLED sid=%s", sid)
        return {"queued": False, "reason": "disabled"}

    runs_root_path = Path(runs_root)
    marker_path = _pipeline_marker_path(runs_root_path, sid)
    if marker_path.exists():
        logger.info("AUTO_AI_SKIP_IN_PROGRESS sid=%s marker=%s", sid, marker_path)
        return {"queued": False, "reason": "in_progress"}

    if not has_ai_merge_best_pairs(sid, runs_root_path):
        logger.info("AUTO_AI_SKIP_NO_CANDIDATES sid=%s", sid)
        return {"queued": False, "reason": "no_candidates"}

    run_dir = runs_root_path / sid
    accounts_dir = run_dir / "cases" / "accounts"

    marker_payload = {"sid": sid, "runs_root": str(runs_root_path)}
    try:
        marker_path.parent.mkdir(parents=True, exist_ok=True)
        marker_path.write_text(
            json.dumps(marker_payload, ensure_ascii=False), encoding="utf-8"
        )
    except OSError:  # pragma: no cover - defensive logging
        logger.warning("AUTO_AI_MARKER_WRITE_FAILED sid=%s path=%s", sid, marker_path)

    logger.info(
        "AUTO_AI_QUEUING sid=%s runs_root=%s accounts_dir=%s marker=%s",
        sid,
        runs_root_path,
        accounts_dir,
        marker_path,
    )

    try:
        from backend.pipeline import auto_ai_tasks

        task_id = auto_ai_tasks.enqueue_auto_ai_chain(sid)
    except Exception:  # pragma: no cover - defensive logging
        logger.error("AUTO_AI_QUEUE_FAILED sid=%s", sid, exc_info=True)
        try:
            if marker_path.exists():
                marker_path.unlink()
        except OSError:
            logger.warning(
                "AUTO_AI_MARKER_CLEANUP_FAILED sid=%s path=%s", sid, marker_path
            )
        raise

    logger.info("AUTO_AI_QUEUED sid=%s", sid)
    payload: dict[str, object] = {"queued": True}
    if task_id:
        payload["task_id"] = task_id
    payload["marker_path"] = str(marker_path)
    return payload


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

