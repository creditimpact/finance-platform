"""Celery tasks implementing the automatic AI adjudication pipeline."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Mapping

from celery import chain, shared_task

from backend.pipeline.auto_ai import (
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


@shared_task(bind=True, autoretry_for=(), retry_backoff=False)
def auto_ai_score_task(self, sid: str, runs_root: str) -> dict[str, object]:
    """Recompute merge scores and persist tags for ``sid``."""

    runs_root_path = Path(runs_root)
    logger.info("AUTO_AI_SCORE start sid=%s", sid)

    try:
        result = score_accounts(sid, runs_root=runs_root_path, write_tags=True)
    except Exception:  # pragma: no cover - defensive logging
        logger.error("AUTO_AI_SCORE failed sid=%s", sid, exc_info=True)
        raise

    touched_accounts = sorted(_normalize_indices(result.indices))
    payload = {
        "sid": sid,
        "runs_root": str(runs_root_path),
        "touched_accounts": touched_accounts,
    }

    logger.info(
        "AUTO_AI_SCORE done sid=%s touched=%d",
        sid,
        len(touched_accounts),
    )
    return payload


@shared_task(bind=True, autoretry_for=(), retry_backoff=False)
def auto_ai_build_packs_task(
    self, prev: Mapping[str, object] | None
) -> dict[str, object]:
    """Build AI merge packs for merge_best AI candidates."""

    payload = _ensure_payload(prev)
    sid = str(payload.get("sid") or "")
    if not sid:
        logger.info("AUTO_AI_BUILD skip: missing sid in payload=%s", payload)
        return payload

    runs_root_value = payload.get("runs_root")
    runs_root_path = Path(str(runs_root_value)) if runs_root_value else Path("runs")
    payload["runs_root"] = str(runs_root_path)

    logger.info("AUTO_AI_BUILD start sid=%s", sid)

    try:
        _build_ai_packs(sid, runs_root_path)
    except Exception:  # pragma: no cover - defensive logging
        logger.error("AUTO_AI_BUILD failed sid=%s", sid, exc_info=True)
        raise

    index_path = runs_root_path / sid / "ai_packs" / "index.json"
    try:
        index_entries = _load_ai_index(index_path)
    except Exception:  # pragma: no cover - defensive logging
        logger.error("AUTO_AI_BUILD invalid index sid=%s path=%s", sid, index_path, exc_info=True)
        raise

    payload["ai_index"] = index_entries

    touched = set()
    for value in payload.get("touched_accounts", []):
        try:
            touched.add(int(value))
        except (TypeError, ValueError):
            continue
    touched.update(_indices_from_index(index_entries))
    payload["touched_accounts"] = sorted(touched)

    logger.info(
        "AUTO_AI_BUILD done sid=%s packs=%d", sid, len(index_entries)
    )
    return payload


@shared_task(bind=True, autoretry_for=(), retry_backoff=False)
def auto_ai_send_packs_task(
    self, prev: Mapping[str, object] | None
) -> dict[str, object]:
    """Send AI packs for adjudication and persist AI decision tags."""

    payload = _ensure_payload(prev)
    sid = str(payload.get("sid") or "")
    if not sid:
        logger.info("AUTO_AI_SEND skip: missing sid in payload=%s", payload)
        return payload

    runs_root_value = payload.get("runs_root")
    runs_root_path = Path(str(runs_root_value)) if runs_root_value else None

    logger.info("AUTO_AI_SEND start sid=%s", sid)

    try:
        _send_ai_packs(sid, runs_root=runs_root_path)
    except Exception:  # pragma: no cover - defensive logging
        logger.error("AUTO_AI_SEND failed sid=%s", sid, exc_info=True)
        raise

    logger.info("AUTO_AI_SEND done sid=%s", sid)
    return payload


@shared_task(bind=True, autoretry_for=(), retry_backoff=False)
def auto_ai_compact_task(
    self, prev: Mapping[str, object] | None, accounts_dir: str
) -> dict[str, object]:
    """Compact tags for accounts touched by the AI pipeline."""

    payload = _ensure_payload(prev)
    indices = payload.get("touched_accounts", [])
    accounts_path = Path(accounts_dir)

    if not accounts_path.exists():
        logger.info(
            "AUTO_AI_COMPACT skip: accounts_dir missing sid=%s dir=%s",
            payload.get("sid"),
            accounts_path,
        )
        return payload

    logger.info(
        "AUTO_AI_COMPACT start sid=%s accounts=%d dir=%s",
        payload.get("sid"),
        len(indices) if isinstance(indices, list) else 0,
        accounts_path,
    )

    try:
        _compact_accounts(accounts_path, indices)
    except Exception:  # pragma: no cover - defensive logging
        logger.error(
            "AUTO_AI_COMPACT failed sid=%s dir=%s",
            payload.get("sid"),
            accounts_path,
            exc_info=True,
        )
        raise

    logger.info("AUTO_AI_COMPACT done sid=%s", payload.get("sid"))
    return payload


@shared_task(bind=True, autoretry_for=(), retry_backoff=False)
def auto_ai_finalize_task(
    self, prev: Mapping[str, object] | None, marker_path: str | None
) -> dict[str, object]:
    """Remove the pipeline-in-progress marker once the workflow completes."""

    payload = _ensure_payload(prev)
    sid = str(payload.get("sid") or "")
    path = Path(marker_path) if marker_path else None

    if path is None:
        logger.info("AUTO_AI_FINALIZE sid=%s marker=None", sid)
        return payload

    try:
        if path.exists():
            path.unlink()
            logger.info("AUTO_AI_FINALIZE sid=%s marker_removed=%s", sid, path)
        else:
            logger.info("AUTO_AI_FINALIZE sid=%s marker_missing=%s", sid, path)
    except Exception:  # pragma: no cover - defensive logging
        logger.warning(
            "AUTO_AI_FINALIZE sid=%s marker_cleanup_failed=%s", sid, path, exc_info=True
        )
    return payload


def enqueue_auto_ai_pipeline(*, sid: str, runs_root: str, marker_path: str | None = None):
    """Queue the Celery task chain for the auto AI adjudication pipeline."""

    runs_root_path = Path(runs_root)
    accounts_dir = runs_root_path / sid / "cases" / "accounts"
    workflow = chain(
        auto_ai_score_task.s(sid, runs_root),
        auto_ai_build_packs_task.s(),
        auto_ai_send_packs_task.s(),
        auto_ai_compact_task.s(str(accounts_dir)),
    )
    if marker_path is not None:
        workflow = chain(workflow, auto_ai_finalize_task.s(marker_path))
    return workflow.apply_async()
