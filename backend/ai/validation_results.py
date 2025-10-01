"""Helpers for ingesting AI validation responses."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from backend.ai.validation_index import ValidationPackIndexWriter
from backend.core.ai.paths import (
    validation_index_path,
    validation_pack_filename_for_account,
    validation_packs_dir,
    validation_result_filename_for_account,
    validation_results_dir,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def _resolve_runs_root(runs_root: Path | str | None) -> Path:
    if runs_root is None:
        return Path("runs").resolve()
    return Path(runs_root).resolve()


def _index_writer(sid: str, runs_root: Path) -> ValidationPackIndexWriter:
    index_path = validation_index_path(sid, runs_root=runs_root, create=True)
    return ValidationPackIndexWriter(sid=sid, index_path=index_path)


def mark_validation_pack_sent(
    sid: str,
    account_id: int | str,
    *,
    runs_root: Path | str | None = None,
    request_lines: int | None = None,
    model: str | None = None,
) -> dict[str, object] | None:
    """Mark the pack for ``account_id`` as sent in the validation index."""

    runs_root_path = _resolve_runs_root(runs_root)
    packs_dir = validation_packs_dir(sid, runs_root=runs_root_path, create=True)
    pack_filename = validation_pack_filename_for_account(account_id)
    pack_path = packs_dir / pack_filename
    writer = _index_writer(sid, runs_root_path)
    return writer.mark_sent(
        pack_path,
        request_lines=request_lines,
        model=model,
    )


def _normalize_result_payload(
    sid: str,
    account_id: int | str,
    payload: Mapping[str, Any],
    *,
    status: str,
    request_lines: int | None,
    model: str | None,
    error: str | None,
    completed_at: str | None,
) -> dict[str, Any]:
    normalized = dict(payload)
    normalized["sid"] = sid
    try:
        normalized["account_id"] = int(account_id)
    except (TypeError, ValueError):
        normalized["account_id"] = account_id

    if request_lines is not None:
        try:
            normalized["request_lines"] = int(request_lines)
        except (TypeError, ValueError):
            normalized.pop("request_lines", None)

    if model is not None:
        normalized["model"] = str(model)
    elif "model" in normalized and normalized["model"] is not None:
        normalized["model"] = str(normalized["model"])

    normalized["status"] = status

    if status == "error" and error:
        normalized.setdefault("error", str(error))

    timestamp = completed_at or normalized.get("completed_at")
    if not isinstance(timestamp, str) or not timestamp.strip():
        normalized["completed_at"] = _utc_now()
    else:
        normalized["completed_at"] = timestamp

    return normalized


def store_validation_result(
    sid: str,
    account_id: int | str,
    response_payload: Mapping[str, Any],
    *,
    runs_root: Path | str | None = None,
    status: str = "done",
    error: str | None = None,
    request_lines: int | None = None,
    model: str | None = None,
    completed_at: str | None = None,
) -> Path:
    """Persist the AI response for ``account_id`` and update the index."""

    normalized_status = str(status).strip().lower()
    if normalized_status not in {"done", "error"}:
        raise ValueError("status must be 'done' or 'error'")

    runs_root_path = _resolve_runs_root(runs_root)
    results_dir = validation_results_dir(sid, runs_root=runs_root_path, create=True)
    result_filename = validation_result_filename_for_account(account_id)
    result_path = results_dir / result_filename

    normalized_payload = _normalize_result_payload(
        sid,
        account_id,
        response_payload,
        status=normalized_status,
        request_lines=request_lines,
        model=model,
        error=error,
        completed_at=completed_at,
    )

    serialized = json.dumps(normalized_payload, ensure_ascii=False, sort_keys=True)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(serialized + "\n", encoding="utf-8")

    packs_dir = validation_packs_dir(sid, runs_root=runs_root_path, create=True)
    pack_filename = validation_pack_filename_for_account(account_id)
    pack_path = packs_dir / pack_filename
    writer = _index_writer(sid, runs_root_path)
    writer.record_result(
        pack_path,
        status=normalized_status,
        error=error,
        request_lines=request_lines,
        model=normalized_payload.get("model"),
        completed_at=normalized_payload.get("completed_at"),
    )

    return result_path


__all__ = ["mark_validation_pack_sent", "store_validation_result"]

