"""Helpers for persisting note_style model outputs."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence

from backend.ai.note_style_results import (
    complete_note_style_result,
    store_note_style_result,
)
from backend.core.ai.paths import NoteStyleAccountPaths
from backend.note_style.validator import coerce_text, validate_analysis_payload
from backend.runflow.manifest import update_note_style_stage_status


log = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _extract_response_content(response_payload: Any) -> str:
    choices: Sequence[Any] | None = None
    if hasattr(response_payload, "choices"):
        choices = getattr(response_payload, "choices")
    elif isinstance(response_payload, Mapping):
        choices = response_payload.get("choices")  # type: ignore[assignment]

    if not isinstance(choices, Sequence) or not choices:
        raise ValueError("Model response missing choices")

    first = choices[0]
    message: Any
    if hasattr(first, "message"):
        message = getattr(first, "message")
    elif isinstance(first, Mapping):
        message = first.get("message")
    else:
        raise ValueError("Model response missing message")

    content: Any
    if hasattr(message, "content"):
        content = getattr(message, "content")
    elif isinstance(message, Mapping):
        content = message.get("content")
    else:
        raise ValueError("Model response missing content")

    if isinstance(content, str):
        text = content
    elif isinstance(content, Sequence):
        pieces: list[str] = []
        for chunk in content:
            if isinstance(chunk, str):
                pieces.append(chunk)
            elif isinstance(chunk, Mapping):
                text_piece = chunk.get("text")
                if isinstance(text_piece, str):
                    pieces.append(text_piece)
        text = "".join(pieces)
    else:
        text = str(content)

    if not text or not text.strip():
        raise ValueError("Model response missing content")

    return text.strip()


def _parse_response_payload(response_payload: Any) -> Mapping[str, Any]:
    text = _extract_response_content(response_payload)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("Model response is not valid JSON") from exc

    if isinstance(parsed, Mapping):
        return parsed

    raise ValueError("Model response payload must be an object")


def ingest_note_style_result(
    *,
    sid: str,
    account_id: str,
    runs_root: Path,
    account_paths: NoteStyleAccountPaths,
    pack_payload: Mapping[str, Any],
    response_payload: Any,
) -> Path:
    """Persist the normalized ``response_payload`` for ``account_id``."""

    parsed = _parse_response_payload(response_payload)
    analysis_payload: Mapping[str, Any]
    if isinstance(parsed.get("analysis"), Mapping):
        analysis_payload = parsed["analysis"]  # type: ignore[assignment]
    else:
        analysis_payload = parsed

    normalized_analysis = validate_analysis_payload(analysis_payload)

    result_payload: MutableMapping[str, Any] = {
        "sid": sid,
        "account_id": str(account_id),
        "analysis": normalized_analysis,
    }

    metrics_payload: MutableMapping[str, Any] | None = None
    note_candidate: Any = None
    if isinstance(pack_payload, Mapping):
        note_candidate = pack_payload.get("note_text")
        if not isinstance(note_candidate, str):
            context = pack_payload.get("context")
            if isinstance(context, Mapping):
                context_note = context.get("note_text")
                if isinstance(context_note, str):
                    note_candidate = context_note
    if isinstance(note_candidate, str):
        metrics_payload = {
            "char_len": len(note_candidate),
            "word_len": len(note_candidate.split()),
        }

    if metrics_payload is not None:
        result_payload["note_metrics"] = metrics_payload

    log.info("NOTE_STYLE_PARSED sid=%s account_id=%s", sid, account_id)

    completed_at = _now_iso()
    result_path = store_note_style_result(
        sid,
        account_id,
        result_payload,
        runs_root=runs_root,
        completed_at=completed_at,
        update_index=False,
    )

    _, totals, _, analysis_valid = complete_note_style_result(
        sid,
        account_id,
        runs_root=runs_root,
        account_paths=account_paths,
        completed_at=completed_at,
    )

    results_completed = int(totals.get("completed", 0)) if totals else 0
    results_failed = int(totals.get("failed", 0)) if totals else 0
    results_count = results_completed + results_failed

    if analysis_valid and results_count > 0:
        try:
            update_note_style_stage_status(
                sid,
                runs_root=runs_root,
                sent=True,
                completed_at=completed_at,
            )
        except Exception:  # pragma: no cover - defensive logging
            log.warning(
                "NOTE_STYLE_MANIFEST_STAGE_STATUS_UPDATE_FAILED sid=%s account_id=%s path=%s",
                sid,
                account_id,
                str(result_path),
                exc_info=True,
            )
        else:
            log.info(
                "NOTE_STYLE_MANIFEST_STAGE_STATUS_UPDATED sid=%s account_id=%s results=%d",
                sid,
                account_id,
                results_count,
            )

    return result_path


__all__ = ["ingest_note_style_result"]
