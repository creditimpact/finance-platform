"""Celery task chain used by the automatic AI adjudication pipeline."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Mapping, MutableMapping

from celery import chain, shared_task

from backend.pipeline.auto_ai import (
    PIPELINE_MARKER_FILENAME,
    _build_ai_packs,
    _compact_accounts,
    _indices_from_index,
    _load_ai_index,
    _normalize_indices,
    _send_ai_packs,
)
from scripts.score_bureau_pairs import score_accounts

logger = logging.getLogger(__name__)


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

    marker_dir = Path("runs") / sid / "ai_packs"
    marker_path = marker_dir / PIPELINE_MARKER_FILENAME
    if marker_path.exists():
        try:
            data = json.loads(marker_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            logger.debug(
                "AUTO_AI_MARKER_READ_FAILED sid=%s path=%s", sid, marker_path, exc_info=True
            )
        else:
            marker_root = data.get("runs_root")
            if isinstance(marker_root, str) and marker_root:
                return Path(marker_root)
        return marker_dir.parent.parent

    return Path("runs")


def _populate_common_paths(payload: MutableMapping[str, object]) -> None:
    sid = str(payload.get("sid") or "")
    if not sid:
        return

    runs_root = _resolve_runs_root(payload, sid)
    accounts_dir = runs_root / sid / "cases" / "accounts"
    marker_path = runs_root / sid / "ai_packs" / PIPELINE_MARKER_FILENAME

    payload["runs_root"] = str(runs_root)
    payload["accounts_dir"] = str(accounts_dir)
    payload["marker_path"] = str(marker_path)


@shared_task(bind=True, autoretry_for=(), retry_backoff=False)
def ai_score_step(self, sid: str) -> dict[str, object]:
    """Recompute merge scores and persist merge tags for ``sid``."""

    logger.info("AUTO_AI_SCORE_START sid=%s", sid)

    payload: dict[str, object] = {"sid": sid}
    _populate_common_paths(payload)

    runs_root = Path(payload["runs_root"])

    try:
        result = score_accounts(sid, runs_root=runs_root, write_tags=True)
    except Exception:  # pragma: no cover - defensive logging
        logger.error("AUTO_AI_SCORE_FAILED sid=%s", sid, exc_info=True)
        raise

    touched_accounts = sorted(_normalize_indices(result.indices))
    payload["touched_accounts"] = touched_accounts

    logger.info(
        "AUTO_AI_SCORE_END sid=%s touched=%d", sid, len(touched_accounts)
    )
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

    logger.info("AUTO_AI_BUILD_START sid=%s", sid)

    try:
        _build_ai_packs(sid, runs_root)
    except Exception:  # pragma: no cover - defensive logging
        logger.error("AUTO_AI_BUILD_FAILED sid=%s", sid, exc_info=True)
        raise

    index_path = runs_root / sid / "ai_packs" / "index.json"
    try:
        index_entries = _load_ai_index(index_path)
    except Exception:  # pragma: no cover - defensive logging
        logger.error("AUTO_AI_BUILD_INVALID_INDEX sid=%s path=%s", sid, index_path, exc_info=True)
        raise

    payload["ai_index"] = index_entries

    touched: set[int] = set(_normalize_indices(payload.get("touched_accounts", [])))
    touched.update(_indices_from_index(index_entries))
    payload["touched_accounts"] = sorted(touched)

    logger.info("AUTO_AI_BUILD_END sid=%s packs=%d", sid, len(index_entries))
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

    logger.info("AUTO_AI_SEND_START sid=%s", sid)

    try:
        _send_ai_packs(sid, runs_root=runs_root)
    except Exception:  # pragma: no cover - defensive logging
        logger.error("AUTO_AI_SEND_FAILED sid=%s", sid, exc_info=True)
        raise

    logger.info("AUTO_AI_SEND_END sid=%s", sid)
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

    logger.info(
        "AUTO_AI_COMPACT_START sid=%s accounts=%d", sid, len(indices)
    )

    if accounts_dir and accounts_dir.exists() and indices:
        try:
            _compact_accounts(accounts_dir, indices)
        except Exception:  # pragma: no cover - defensive logging
            logger.error(
                "AUTO_AI_COMPACT_FAILED sid=%s dir=%s", sid, accounts_dir, exc_info=True
            )
            raise

    logger.info("AUTO_AI_COMPACT_END sid=%s", sid)

    marker_value = payload.get("marker_path")
    if marker_value:
        marker_path = Path(str(marker_value))
        try:
            if marker_path.exists():
                marker_path.unlink()
                logger.info(
                    "AUTO_AI_CHAIN_DONE sid=%s marker_removed=%s", sid, marker_path
                )
            else:
                logger.info(
                    "AUTO_AI_CHAIN_DONE sid=%s marker_missing=%s", sid, marker_path
                )
        except Exception:  # pragma: no cover - defensive logging
            logger.warning(
                "AUTO_AI_CHAIN_DONE sid=%s marker_cleanup_failed=%s",
                sid,
                marker_path,
                exc_info=True,
            )
    else:
        logger.info("AUTO_AI_CHAIN_DONE sid=%s marker_missing=None", sid)

    return payload


def enqueue_auto_ai_chain(sid: str) -> str:
    """Queue the AI adjudication Celery chain and return the root task id."""

    logger.info("AUTO_AI_CHAIN_START sid=%s", sid)

    workflow = chain(
        ai_score_step.s(sid),
        ai_build_packs_step.s(),
        ai_send_packs_step.s(),
        ai_compact_tags_step.s(),
    )

    result = workflow.apply_async()
    task_id = str(result.id)

    logger.info("AUTO_AI_CHAIN_ENQUEUED sid=%s task_id=%s", sid, task_id)
    return task_id

