"""Celery task chain used by the automatic AI adjudication pipeline."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, MutableMapping

from celery import chain, shared_task

from backend.pipeline.auto_ai import (
    AUTO_AI_PIPELINE_DIRNAME,
    INFLIGHT_LOCK_FILENAME,
    LAST_OK_FILENAME,
    packs_dir_for,
    _build_ai_packs,
    _compact_accounts,
    _indices_from_index,
    _load_ai_index,
    _normalize_indices,
    _send_ai_packs,
    has_ai_merge_best_pairs,
)
from scripts.score_bureau_pairs import score_accounts

LEGACY_PIPELINE_DIRNAME = "ai_packs"
LEGACY_MARKER_FILENAME = "auto_ai_pipeline_in_progress.json"

logger = logging.getLogger(__name__)


def _append_run_log_entry(
    *,
    runs_root: Path,
    sid: str,
    packs: int,
    pairs: int,
    reason: str | None = None,
) -> None:
    """Append a compact JSON line describing the AI run outcome."""

    logs_path = packs_dir_for(sid, runs_root=runs_root) / "logs.txt"
    entry = {
        "sid": sid,
        "at": datetime.now(timezone.utc).isoformat(),
        "packs": int(packs),
        "pairs": int(pairs),
    }
    if reason:
        entry["reason"] = reason

    try:
        logs_path.parent.mkdir(parents=True, exist_ok=True)
        with logs_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:  # pragma: no cover - defensive logging
        logger.warning(
            "AUTO_AI_LOG_APPEND_FAILED sid=%s path=%s", sid, logs_path, exc_info=True
        )



def _ensure_payload(prev: Mapping[str, object] | None) -> dict[str, object]:
    if isinstance(prev, Mapping):
        return dict(prev)
    return {}


def _resolve_runs_root(payload: Mapping[str, object], sid: str) -> Path:
    runs_root_value = payload.get("runs_root")
    if isinstance(runs_root_value, (str, os.PathLike)):
        return Path(runs_root_value)

    env_root = os.environ.get("RUNS_ROOT")
    if env_root:
        return Path(env_root)

    default_root = Path("runs")

    pipeline_dir = packs_dir_for(sid, runs_root=default_root)
    lock_path = pipeline_dir / INFLIGHT_LOCK_FILENAME
    if lock_path.exists():
        try:
            data = json.loads(lock_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            logger.debug(
                "AUTO_AI_LOCK_READ_FAILED sid=%s path=%s", sid, lock_path, exc_info=True
            )
        else:
            marker_root = data.get("runs_root")
            if isinstance(marker_root, str) and marker_root:
                return Path(marker_root)
        return pipeline_dir.parent.parent

    legacy_dir = default_root / sid / LEGACY_PIPELINE_DIRNAME
    legacy_marker = legacy_dir / LEGACY_MARKER_FILENAME
    if legacy_marker.exists():
        try:
            data = json.loads(legacy_marker.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            logger.debug(
                "AUTO_AI_LEGACY_MARKER_READ_FAILED sid=%s path=%s",
                sid,
                legacy_marker,
                exc_info=True,
            )
        else:
            marker_root = data.get("runs_root")
            if isinstance(marker_root, str) and marker_root:
                return Path(marker_root)
        return legacy_dir.parent.parent

    return default_root


def _populate_common_paths(payload: MutableMapping[str, object]) -> None:
    sid = str(payload.get("sid") or "")
    if not sid:
        return

    runs_root = _resolve_runs_root(payload, sid)
    accounts_dir = runs_root / sid / "cases" / "accounts"
    pipeline_dir = packs_dir_for(sid, runs_root=runs_root)
    lock_path = pipeline_dir / INFLIGHT_LOCK_FILENAME
    last_ok_path = pipeline_dir / LAST_OK_FILENAME

    payload["runs_root"] = str(runs_root)
    payload["accounts_dir"] = str(accounts_dir)
    payload["pipeline_dir"] = str(pipeline_dir)
    payload["lock_path"] = str(lock_path)
    payload["marker_path"] = str(lock_path)
    payload["last_ok_path"] = str(last_ok_path)


def _cleanup_lock(payload: Mapping[str, object], *, reason: str) -> bool:
    sid = str(payload.get("sid") or "")
    lock_value = payload.get("lock_path") or payload.get("marker_path")
    if not lock_value:
        return False

    lock_path = Path(str(lock_value))
    try:
        if lock_path.exists():
            lock_path.unlink()
            logger.info(
                "AUTO_AI_LOCK_REMOVED sid=%s reason=%s lock=%s",
                sid,
                reason,
                lock_path,
            )
            return True
        logger.info(
            "AUTO_AI_LOCK_MISSING sid=%s reason=%s lock=%s",
            sid,
            reason,
            lock_path,
        )
        return False
    except Exception:  # pragma: no cover - defensive logging
        logger.warning(
            "AUTO_AI_LOCK_CLEANUP_FAILED sid=%s reason=%s lock=%s",
            sid,
            reason,
            lock_path,
            exc_info=True,
        )
        return False


@shared_task(bind=True, autoretry_for=(), retry_backoff=False)
def ai_score_step(self, sid: str, runs_root: str | None = None) -> dict[str, object]:
    """Recompute merge scores and persist merge tags for ``sid``."""

    logger.info("AI_SCORE_START sid=%s", sid)

    payload: dict[str, object] = {"sid": sid}
    if runs_root is not None:
        payload["runs_root"] = runs_root
    _populate_common_paths(payload)

    runs_root = Path(payload["runs_root"])

    try:
        result = score_accounts(sid, runs_root=runs_root, write_tags=True)
    except Exception:  # pragma: no cover - defensive logging
        logger.error("AUTO_AI_SCORE_FAILED sid=%s", sid, exc_info=True)
        _cleanup_lock(payload, reason="score_failed")
        raise

    touched_accounts = sorted(_normalize_indices(result.indices))
    payload["touched_accounts"] = touched_accounts

    logger.info("AI_SCORE_END sid=%s touched=%d", sid, len(touched_accounts))
    return payload


@shared_task(bind=True, autoretry_for=(), retry_backoff=False)
def ai_build_packs_step(self, prev: Mapping[str, object] | None) -> dict[str, object]:
    """Build AI merge packs for accounts requiring AI decisions."""

    payload = _ensure_payload(prev)
    sid = str(payload.get("sid") or "")
    if not sid:
        logger.info("AUTO_AI_BUILD_SKIP payload=%s", payload)
        return payload

    _populate_common_paths(payload)
    runs_root = Path(payload["runs_root"])

    logger.info("AI_BUILD_START sid=%s", sid)

    if not has_ai_merge_best_pairs(sid, runs_root):
        logger.info("AUTO_AI_SKIP_NO_CANDIDATES sid=%s", sid)
        payload["ai_index"] = []
        payload["skip_reason"] = "no_candidates"
        return payload

    try:
        _build_ai_packs(sid, runs_root)
    except Exception:  # pragma: no cover - defensive logging
        logger.error("AUTO_AI_BUILD_FAILED sid=%s", sid, exc_info=True)
        _cleanup_lock(payload, reason="build_failed")
        raise

    index_path = packs_dir_for(sid, runs_root=runs_root) / "index.json"
    try:
        index_entries = _load_ai_index(index_path)
    except Exception:  # pragma: no cover - defensive logging
        logger.error("AUTO_AI_BUILD_INVALID_INDEX sid=%s path=%s", sid, index_path, exc_info=True)
        _cleanup_lock(payload, reason="build_invalid_index")
        raise

    payload["ai_index"] = index_entries

    touched: set[int] = set(_normalize_indices(payload.get("touched_accounts", [])))
    touched.update(_indices_from_index(index_entries))
    payload["touched_accounts"] = sorted(touched)

    logger.info("AI_PACKS_INDEX sid=%s path=%s count=%d", sid, index_path, len(index_entries))
    logger.info("AI_BUILD_END sid=%s packs=%d", sid, len(index_entries))
    return payload


@shared_task(bind=True, autoretry_for=(), retry_backoff=False)
def ai_send_packs_step(self, prev: Mapping[str, object] | None) -> dict[str, object]:
    """Send AI merge packs for adjudication and persist AI decision tags."""

    payload = _ensure_payload(prev)
    sid = str(payload.get("sid") or "")
    if not sid:
        logger.info("AUTO_AI_SEND_SKIP payload=%s", payload)
        return payload

    _populate_common_paths(payload)
    runs_root = Path(payload["runs_root"])

    logger.info("AI_SEND_START sid=%s", sid)

    index_entries = payload.get("ai_index")
    if not index_entries:
        reason = str(payload.get("skip_reason") or "no_packs")
        logger.info("AI_SEND_SKIP sid=%s reason=%s", sid, reason)
        return payload

    try:
        _send_ai_packs(sid, runs_root=runs_root)
    except Exception:  # pragma: no cover - defensive logging
        logger.error("AUTO_AI_SEND_FAILED sid=%s", sid, exc_info=True)
        _cleanup_lock(payload, reason="send_failed")
        raise

    logger.info("AI_SEND_END sid=%s", sid)
    return payload


@shared_task(bind=True, autoretry_for=(), retry_backoff=False)
def ai_compact_tags_step(self, prev: Mapping[str, object] | None) -> dict[str, object]:
    """Compact tags and summaries for accounts touched by the AI pipeline."""

    payload = _ensure_payload(prev)
    sid = str(payload.get("sid") or "")
    if not sid:
        logger.info("AUTO_AI_COMPACT_SKIP payload=%s", payload)
        return payload

    _populate_common_paths(payload)

    accounts_dir_value = payload.get("accounts_dir")
    accounts_dir = Path(str(accounts_dir_value)) if accounts_dir_value else None
    indices = sorted(_normalize_indices(payload.get("touched_accounts", [])))

    logger.info("AI_COMPACT_START sid=%s accounts=%d", sid, len(indices))

    if accounts_dir and accounts_dir.exists() and indices:
        try:
            _compact_accounts(accounts_dir, indices)
        except Exception:  # pragma: no cover - defensive logging
            logger.error(
                "AUTO_AI_COMPACT_FAILED sid=%s dir=%s", sid, accounts_dir, exc_info=True
            )
            _cleanup_lock(payload, reason="compact_failed")
            raise

    logger.info("AI_COMPACT_END sid=%s", sid)

    packs_count = len(payload.get("ai_index", []) or [])
    pairs_count = len(indices)
    payload["packs"] = packs_count
    payload["pairs"] = pairs_count

    last_ok_value = payload.get("last_ok_path")
    if last_ok_value:
        last_ok_path = Path(str(last_ok_value))
        last_ok_payload = {
            "sid": sid,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "packs": packs_count,
            "pairs": pairs_count,
        }
        try:
            last_ok_path.write_text(
                json.dumps(last_ok_payload, ensure_ascii=False), encoding="utf-8"
            )
            logger.info("AUTO_AI_LAST_OK sid=%s path=%s", sid, last_ok_path)
        except Exception:  # pragma: no cover - defensive logging
            logger.warning(
                "AUTO_AI_LAST_OK_WRITE_FAILED sid=%s path=%s",
                sid,
                last_ok_path,
                exc_info=True,
            )

    runs_root_value = payload.get("runs_root")
    if runs_root_value:
        try:
            runs_root_path = Path(str(runs_root_value))
        except Exception:  # pragma: no cover - defensive logging
            logger.debug(
                "AUTO_AI_LOG_ROOT_INVALID sid=%s runs_root=%r",
                sid,
                runs_root_value,
                exc_info=True,
            )
        else:
            skip_reason = payload.get("skip_reason")
            reason_text = str(skip_reason) if isinstance(skip_reason, str) else None
            _append_run_log_entry(
                runs_root=runs_root_path,
                sid=sid,
                packs=packs_count,
                pairs=pairs_count,
                reason=reason_text,
            )

    removed = _cleanup_lock(payload, reason="chain_complete")
    logger.info(
        "AUTO_AI_CHAIN_END sid=%s lock_removed=%s packs=%s pairs=%s",
        sid,
        1 if removed else 0,
        packs_count,
        pairs_count,
    )

    return payload


def enqueue_auto_ai_chain(sid: str, runs_root: Path | str | None = None) -> str:
    """Queue the AI adjudication Celery chain and return the root task id."""

    runs_root_value = str(runs_root) if runs_root is not None else None

    logger.info("AUTO_AI_CHAIN_START sid=%s runs_root=%s", sid, runs_root_value)

    workflow = chain(
        ai_score_step.s(sid, runs_root_value),
        ai_build_packs_step.s(),
        ai_send_packs_step.s(),
        ai_compact_tags_step.s(),
    )

    result = workflow.apply_async()
    task_id = str(result.id)

    logger.info(
        "AUTO_AI_CHAIN_ENQUEUED sid=%s task_id=%s runs_root=%s",
        sid,
        task_id,
        runs_root_value,
    )
    return task_id

