"""Automatic AI adjudication hooks for the case-build pipeline."""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from backend.core.io.tags_minimize import compact_account_tags
from backend.pipeline.runs import RunManifest
from scripts.build_ai_merge_packs import main as build_ai_merge_packs_main
from scripts.send_ai_merge_packs import main as send_ai_merge_packs_main

logger = logging.getLogger(__name__)

AUTO_AI_PIPELINE_DIRNAME = ".ai_pipeline"
INFLIGHT_LOCK_FILENAME = "inflight.lock"
LAST_OK_FILENAME = "last_ok.json"
DEFAULT_INFLIGHT_TTL_SECONDS = 30 * 60


def _pipeline_dir(runs_root: Path, sid: str) -> Path:
    return runs_root / sid / AUTO_AI_PIPELINE_DIRNAME


def _inflight_lock_path(runs_root: Path, sid: str) -> Path:
    return _pipeline_dir(runs_root, sid) / INFLIGHT_LOCK_FILENAME


def _last_ok_path(runs_root: Path, sid: str) -> Path:
    return _pipeline_dir(runs_root, sid) / LAST_OK_FILENAME


def _lock_age_seconds(path: Path, *, now: float | None = None) -> float | None:
    try:
        stat = path.stat()
    except OSError:
        return None
    reference = now if now is not None else time.time()
    return max(0.0, reference - stat.st_mtime)


def _lock_is_stale(
    path: Path,
    *,
    ttl_seconds: int,
    now: float | None = None,
) -> bool:
    if ttl_seconds <= 0:
        return True
    age = _lock_age_seconds(path, now=now)
    if age is None:
        return True
    return age >= ttl_seconds


def maybe_run_auto_ai_pipeline(
    sid: str,
    *,
    summary: Mapping[str, object] | None = None,
    force: bool = False,
    inflight_ttl_seconds: int = DEFAULT_INFLIGHT_TTL_SECONDS,
    now: float | None = None,
) -> dict[str, object]:
    """Backward-compatible shim that queues the auto-AI pipeline."""

    _ = summary  # preserved for compatibility with older call sites
    manifest = RunManifest.for_sid(sid)
    runs_root = manifest.path.parent.parent
    return maybe_queue_auto_ai_pipeline(
        sid,
        runs_root=runs_root,
        flag_env=os.environ,
        force=force,
        inflight_ttl_seconds=inflight_ttl_seconds,
        now=now,
    )


def maybe_queue_auto_ai_pipeline(
    sid: str,
    *,
    runs_root: Path,
    flag_env: Mapping[str, str],
    force: bool = False,
    inflight_ttl_seconds: int = DEFAULT_INFLIGHT_TTL_SECONDS,
    now: float | None = None,
) -> dict[str, object]:
    """Queue the automatic AI adjudication pipeline when enabled."""

    flag_value = str(flag_env.get("ENABLE_AUTO_AI_PIPELINE", "0"))
    if flag_value != "1":
        logger.info("AUTO_AI_SKIP_DISABLED sid=%s", sid)
        return {"queued": False, "reason": "disabled"}

    runs_root_path = Path(runs_root)
    lock_path = _inflight_lock_path(runs_root_path, sid)
    pipeline_dir = lock_path.parent

    lock_exists = lock_path.exists()
    lock_age = _lock_age_seconds(lock_path, now=now) if lock_exists else None
    lock_stale = lock_exists and _lock_is_stale(
        lock_path, ttl_seconds=inflight_ttl_seconds, now=now
    )

    if lock_exists and not (lock_stale or force):
        logger.info(
            "AUTO_AI_SKIP_INFLIGHT sid=%s lock=%s age=%s ttl=%s",
            sid,
            lock_path,
            f"{lock_age:.1f}" if lock_age is not None else "unknown",
            inflight_ttl_seconds,
        )
        return {"queued": False, "reason": "inflight"}

    if lock_exists and (lock_stale or force):
        logger.info(
            "AUTO_AI_LOCK_CLEAR sid=%s lock=%s stale=%s force=%s age=%s",
            sid,
            lock_path,
            1 if lock_stale else 0,
            1 if force else 0,
            f"{lock_age:.1f}" if lock_age is not None else "unknown",
        )
        try:
            lock_path.unlink()
        except OSError:  # pragma: no cover - defensive logging
            logger.warning(
                "AUTO_AI_LOCK_CLEAR_FAILED sid=%s lock=%s", sid, lock_path, exc_info=True
            )

    if not has_ai_merge_best_pairs(sid, runs_root_path):
        logger.info("AUTO_AI_SKIP_NO_CANDIDATES sid=%s", sid)
        return {"queued": False, "reason": "no_candidates"}

    run_dir = runs_root_path / sid
    accounts_dir = run_dir / "cases" / "accounts"

    lock_payload = {
        "sid": sid,
        "runs_root": str(runs_root_path),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    if force:
        lock_payload["force"] = True

    try:
        pipeline_dir.mkdir(parents=True, exist_ok=True)
        lock_path.write_text(json.dumps(lock_payload, ensure_ascii=False), encoding="utf-8")
    except OSError:  # pragma: no cover - defensive logging
        logger.warning("AUTO_AI_LOCK_WRITE_FAILED sid=%s path=%s", sid, lock_path)

    logger.info(
        "AUTO_AI_QUEUING sid=%s runs_root=%s accounts_dir=%s lock=%s",
        sid,
        runs_root_path,
        accounts_dir,
        lock_path,
    )

    try:
        from backend.pipeline import auto_ai_tasks

        task_id = auto_ai_tasks.enqueue_auto_ai_chain(sid, runs_root=runs_root_path)
    except Exception:  # pragma: no cover - defensive logging
        logger.error("AUTO_AI_QUEUE_FAILED sid=%s", sid, exc_info=True)
        try:
            if lock_path.exists():
                lock_path.unlink()
        except OSError:
            logger.warning(
                "AUTO_AI_LOCK_CLEANUP_FAILED sid=%s path=%s", sid, lock_path
            )
        raise

    logger.info("AUTO_AI_QUEUED sid=%s", sid)
    payload: dict[str, object] = {"queued": True, "reason": "queued"}
    if task_id:
        payload["task_id"] = task_id
    payload["lock_path"] = str(lock_path)
    payload["pipeline_dir"] = str(pipeline_dir)
    payload["last_ok_path"] = str(_last_ok_path(runs_root_path, sid))
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
            compact_account_tags(account_dir)
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

