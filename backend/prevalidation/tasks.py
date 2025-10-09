from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Mapping

from celery import shared_task

from backend.config import PREVALIDATION_DETECT_DATES
from backend.prevalidation.date_convention_detector import detect_month_language_for_run
from backend.prevalidation.io_utils import atomic_merge_json

logger = logging.getLogger(__name__)


def _ensure_payload(prev: Mapping[str, object] | None) -> dict[str, object]:
    if isinstance(prev, Mapping):
        return dict(prev)
    return {}


def _resolve_runs_root(payload: Mapping[str, object], sid: str) -> Path:
    runs_root_value = payload.get("runs_root")
    if isinstance(runs_root_value, (str, os.PathLike)):
        return Path(runs_root_value)

    env_root = os.getenv("RUNS_ROOT")
    if env_root:
        return Path(env_root)

    return Path("runs")


def _resolve_general_info_path(run_root: Path) -> Path | None:
    manifest_path = run_root / "manifest.json"
    if manifest_path.exists():
        try:
            manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            logger.debug(
                "DATE_CONVENTION_MANIFEST_READ_FAILED run_root=%s path=%s",
                run_root,
                manifest_path,
                exc_info=True,
            )
        else:
            general_value = (
                manifest_data.get("artifacts", {})
                .get("traces", {})
                .get("accounts_table", {})
                .get("general_json")
            )
            if isinstance(general_value, str) and general_value.strip():
                general_path = Path(general_value)
                if general_path.exists():
                    return general_path

    default_path = run_root / "traces" / "accounts_table" / "general_info_from_full.json"
    if default_path.exists():
        return default_path
    return None


def _load_general_payload(path: Path) -> dict[str, object] | None:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        logger.warning(
            "DATE_CONVENTION_READ_FAILED path=%s", path, exc_info=True
        )
        return None

    if not raw.strip():
        return {}

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning(
            "DATE_CONVENTION_INVALID_JSON path=%s", path, exc_info=True
        )
        return None

    if not isinstance(payload, dict):
        logger.warning(
            "DATE_CONVENTION_INVALID_TYPE path=%s type=%s",
            path,
            type(payload).__name__,
        )
        return None

    return payload


def detect_and_persist_date_convention(
    sid: str,
    *,
    runs_root: Path | str | None = None,
) -> dict[str, object] | None:
    """Run the month language detector for ``sid`` and persist the result."""

    if not PREVALIDATION_DETECT_DATES:
        logger.info("DATE_CONVENTION_SKIP sid=%s reason=disabled", sid)
        return None

    if not sid:
        return None

    base_root = Path(runs_root) if runs_root is not None else Path(os.getenv("RUNS_ROOT", "runs"))
    run_root = base_root / sid
    if not run_root.exists():
        logger.info("DATE_CONVENTION_SKIP sid=%s reason=run_missing path=%s", sid, run_root)
        return None

    general_info_path = _resolve_general_info_path(run_root)
    if general_info_path is None:
        logger.info(
            "DATE_CONVENTION_SKIP sid=%s reason=general_info_missing run_root=%s",
            sid,
            run_root,
        )
        return None

    payload = _load_general_payload(general_info_path)
    if payload is None:
        logger.info(
            "DATE_CONVENTION_SKIP sid=%s reason=general_info_unreadable path=%s",
            sid,
            general_info_path,
        )
        return None

    existing_block = payload.get("date_convention") if isinstance(payload, dict) else None
    if isinstance(existing_block, dict):
        logger.info("DATE_CONVENTION_SKIP sid=%s reason=already_present", sid)
        return existing_block

    detection = detect_month_language_for_run(str(run_root))
    block = detection.get("date_convention") if isinstance(detection, dict) else None
    if not isinstance(block, dict):
        logger.info("DATE_CONVENTION_SKIP sid=%s reason=no_detection", sid)
        return None

    try:
        atomic_merge_json(str(general_info_path), "date_convention", dict(block))
    except Exception:
        logger.error(
            "DATE_CONVENTION_WRITE_FAILED sid=%s path=%s", sid, general_info_path, exc_info=True
        )
        return None

    evidence = block.get("evidence_counts") if isinstance(block.get("evidence_counts"), dict) else {}
    logger.info(
        "DATE_CONVENTION_WRITTEN sid=%s language=%s he_hits=%s en_hits=%s accounts=%s",
        sid,
        block.get("month_language"),
        evidence.get("he_hits"),
        evidence.get("en_hits"),
        evidence.get("accounts_scanned"),
    )
    return block


@shared_task(bind=True, autoretry_for=(), retry_backoff=False)
def run_date_convention_detector(
    self, prev: Mapping[str, object] | None
) -> dict[str, object]:
    """Celery task wrapper that runs the date convention detector."""

    payload = _ensure_payload(prev)
    sid = str(payload.get("sid") or "")
    if not sid:
        logger.info("DATE_CONVENTION_PAYLOAD_SKIP payload=%s", payload)
        return payload

    runs_root = _resolve_runs_root(payload, sid)
    try:
        block = detect_and_persist_date_convention(sid, runs_root=runs_root)
    except Exception:
        logger.error("DATE_CONVENTION_TASK_FAILED sid=%s", sid, exc_info=True)
        return payload

    if isinstance(block, dict):
        payload["date_convention"] = dict(block)
    return payload


__all__ = [
    "detect_and_persist_date_convention",
    "run_date_convention_detector",
]
