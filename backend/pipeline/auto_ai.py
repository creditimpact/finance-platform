"""Automatic AI adjudication hooks for the case-build pipeline."""

from __future__ import annotations

import glob
import json
import logging
import os
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from celery import shared_task

from backend.core.logic.tags.compact import (
    compact_account_tags,
    compact_tags_for_sid,
)
from backend.pipeline.runs import RUNS_ROOT, RunManifest, persist_manifest
from scripts.build_ai_merge_packs import main as build_ai_merge_packs_main
from scripts.send_ai_merge_packs import main as send_ai_merge_packs_main

logger = logging.getLogger(__name__)

AUTO_AI_PIPELINE_DIRNAME = "ai_packs"
INFLIGHT_LOCK_FILENAME = "inflight.lock"
LAST_OK_FILENAME = "last_ok.json"
DEFAULT_INFLIGHT_TTL_SECONDS = 30 * 60


def packs_dir_for(sid: str, *, runs_root: Path | str | None = None) -> Path:
    """Return the canonical ``ai_packs`` directory for ``sid``."""

    base = Path(runs_root) if runs_root is not None else RUNS_ROOT
    return Path(base) / sid / AUTO_AI_PIPELINE_DIRNAME


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
    packs_dir = packs_dir_for(sid, runs_root=runs_root_path)
    lock_path = packs_dir / INFLIGHT_LOCK_FILENAME
    last_ok_path = packs_dir / LAST_OK_FILENAME

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
        packs_dir.mkdir(parents=True, exist_ok=True)
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
    manifest = RunManifest.for_sid(sid)
    manifest.set_ai_enqueued()
    persist_manifest(manifest)
    logger.info("MANIFEST_AI_ENQUEUED sid=%s", sid)
    payload: dict[str, object] = {"queued": True, "reason": "queued"}
    if task_id:
        payload["task_id"] = task_id
    payload["lock_path"] = str(lock_path)
    payload["pipeline_dir"] = str(packs_dir)
    payload["last_ok_path"] = str(last_ok_path)
    return payload


def has_ai_merge_best_tags(sid: str) -> bool:
    """Return ``True`` when any account tags request AI merge adjudication."""

    acc_glob = Path(RUNS_ROOT) / sid / "cases" / "accounts" / "*" / "tags.json"
    for path in glob.glob(str(acc_glob)):
        try:
            raw = Path(path).read_text(encoding="utf-8")
        except OSError:
            continue
        try:
            payload = json.loads(raw)
        except Exception:
            continue

        if isinstance(payload, list):
            entries = payload
        elif isinstance(payload, Mapping):
            tags = payload.get("tags")
            entries = tags if isinstance(tags, list) else []
        else:
            entries = []

        for tag in entries:
            if isinstance(tag, Mapping):
                kind = str(tag.get("kind", "")).strip().lower()
                decision = str(tag.get("decision", "")).strip().lower()
                if kind == "merge_best" and decision == "ai":
                    return True

    return False


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

    if isinstance(data, Mapping):
        pairs = data.get("pairs")
        if not isinstance(pairs, list):
            return []
        entries: list[Mapping[str, object]] = []
        for entry in pairs:
            if isinstance(entry, Mapping):
                entries.append(entry)
        return entries

    if isinstance(data, list):  # pragma: no cover - legacy support
        entries: list[Mapping[str, object]] = []
        for entry in data:
            if isinstance(entry, Mapping):
                entries.append(entry)
        return entries

    raise ValueError(f"AI pack index must be a list or mapping: {path}")


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

    return True


@contextmanager
def ai_inflight_lock(runs_root: Path, sid: str):
    """
    Prevents concurrent AI runs on the same SID.
    Creates runs/<sid>/ai_packs/inflight.lock; removes it on exit.
    """

    ai_dir = packs_dir_for(sid, runs_root=runs_root)
    ai_dir.mkdir(parents=True, exist_ok=True)
    lock = ai_dir / INFLIGHT_LOCK_FILENAME
    if lock.exists():
        # Someone else is running; caller should skip.
        raise RuntimeError("AI pipeline already inflight")
    lock.write_text("1", encoding="utf-8")
    try:
        yield
    finally:
        try:
            lock.unlink()
        except FileNotFoundError:
            pass


def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default) in ("1", "true", "True")


def maybe_run_ai_pipeline(sid: str) -> dict[str, object]:
    """Queue the lightweight automatic AI pipeline after case building."""

    if not _env_flag("ENABLE_AUTO_AI_PIPELINE", "1"):
        return {"sid": sid, "skipped": "feature_off"}

    if not has_ai_merge_best_tags(sid):
        manifest = RunManifest.for_sid(sid)
        manifest.set_ai_skipped("no_candidates")
        persist_manifest(manifest)
        logger.info("AUTO_AI_SKIPPED sid=%s reason=no_ai_merge_best", sid)
        return {"sid": sid, "skipped": "no_ai_candidates"}

    try:
        with ai_inflight_lock(RUNS_ROOT, sid):
            return _run_auto_ai_pipeline(sid)
    except RuntimeError:
        return {"sid": sid, "skipped": "inflight"}


@shared_task(name="pipeline.maybe_run_ai_pipeline")
def maybe_run_ai_pipeline_task(sid: str):
    """Celery task wrapper that launches the auto-AI pipeline asynchronously."""

    return maybe_run_ai_pipeline(sid)


def _run_auto_ai_pipeline(sid: str):
    # === 1) score → (re)write merge_* tags
    from backend.core.logic.merge.scorer import score_bureau_pairs_cli

    score_bureau_pairs_cli(sid=sid, write_tags=True, runs_root=RUNS_ROOT)

    if not has_ai_merge_best_tags(sid):
        return {"sid": sid, "skipped": "candidates_disappeared_after_score"}

    # === 2) build packs
    _build_ai_packs(sid, RUNS_ROOT)

    manifest = RunManifest.for_sid(sid)
    packs_dir = packs_dir_for(sid, runs_root=RUNS_ROOT)
    index_path = packs_dir / "index.json"

    manifest_pairs = 0
    ai_section = manifest.data.get("ai") if isinstance(manifest.data, dict) else {}
    if isinstance(ai_section, dict):
        packs_section = ai_section.get("packs")
        if isinstance(packs_section, dict):
            try:
                manifest_pairs = int(packs_section.get("pairs") or 0)
            except (TypeError, ValueError):
                manifest_pairs = 0

    if not index_path.exists():
        if manifest_pairs > 0:
            logger.error(
                "AUTO_AI_NO_PACKS_INDEX_MISSING sid=%s manifest_pairs=%d",
                sid,
                manifest_pairs,
            )
            return {"sid": sid, "skipped": "no_packs"}
        manifest.set_ai_skipped("no_packs")
        persist_manifest(manifest)
        logger.info(
            "AUTO_AI_SKIP_NO_PACKS sid=%s packs_dir=%s (index missing)", sid, packs_dir
        )
        return {"sid": sid, "skipped": "no_packs"}

    try:
        index_data = json.loads(index_path.read_text(encoding="utf-8"))
        if not isinstance(index_data, dict):
            index_data = {}
        pairs_count = int(
            index_data.get("pairs_count")
            or len(index_data.get("packs") or [])
        )
    except Exception as exc:
        logger.exception(
            "AUTO_AI_SKIP_NO_PACKS sid=%s reason=index_load_error error=%s", sid, exc
        )
        if manifest_pairs > 0:
            logger.error(
                "AUTO_AI_NO_PACKS_INDEX_ERROR sid=%s manifest_pairs=%d",
                sid,
                manifest_pairs,
            )
            return {"sid": sid, "skipped": "no_packs"}
        manifest.set_ai_skipped("no_packs")
        persist_manifest(manifest)
        return {"sid": sid, "skipped": "no_packs"}

    if pairs_count <= 0:
        if manifest_pairs > 0:
            logger.error(
                "AUTO_AI_NO_PACKS_COUNT_MISMATCH sid=%s manifest_pairs=%d",
                sid,
                manifest_pairs,
            )
            return {"sid": sid, "skipped": "no_packs"}
        manifest.set_ai_skipped("no_packs")
        persist_manifest(manifest)
        logger.info(
            "AUTO_AI_SKIP_NO_PACKS sid=%s packs_dir=%s (pairs_count=0)", sid, packs_dir
        )
        return {"sid": sid, "skipped": "no_packs"}

    logger.info("AUTO_AI_PACKS_FOUND sid=%s dir=%s pairs=%d", sid, packs_dir, pairs_count)
    logger.info("AUTO_AI_BUILT sid=%s pairs=%d", sid, pairs_count)
    manifest = manifest.set_ai_built(packs_dir, pairs_count)
    persist_manifest(manifest)

    # === 3) send to AI (writes ai_decision / same_debt_pair)
    argv = ["--sid", sid, "--packs-dir", str(packs_dir)]
    send_ai_merge_packs_main(argv)

    manifest.set_ai_sent()
    persist_manifest(manifest)
    logger.info("AUTO_AI_SENT sid=%s dir=%s", sid, packs_dir)

    # === 4) compact tags (keep only tags; push explanations to summary.json)
    compact_tags_for_sid(sid)
    manifest.set_ai_compacted()
    persist_manifest(manifest)
    logger.info("AUTO_AI_COMPACTED sid=%s", sid)

    logger.info("AUTO_AI_DONE sid=%s", sid)
    return {"sid": sid, "ok": True}

