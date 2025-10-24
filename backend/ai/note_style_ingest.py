"""Helpers for persisting note_style model outputs."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence

from backend.ai.note_style_results import store_note_style_result
from backend.core.ai.paths import NoteStyleAccountPaths
from backend.note_style.validator import coerce_text, validate_analysis_payload


log = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_existing_payload(path: Path) -> Mapping[str, Any] | None:
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError:
        log.warning("STYLE_INGEST_EXISTING_READ_FAILED path=%s", path, exc_info=True)
        return None

    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    if not lines:
        return None

    try:
        payload = json.loads(lines[-1])
    except json.JSONDecodeError:
        log.warning("STYLE_INGEST_EXISTING_PARSE_FAILED path=%s", path, exc_info=True)
        return None

    return payload if isinstance(payload, Mapping) else None




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

    existing_payload = _load_existing_payload(account_paths.result_file)

    prompt_salt = coerce_text(pack_payload.get("prompt_salt"), preserve_case=True)
    if not prompt_salt and isinstance(existing_payload, Mapping):
        prompt_salt = coerce_text(
            existing_payload.get("prompt_salt"), preserve_case=True
        )

    note_hash = coerce_text(pack_payload.get("note_hash"), preserve_case=True)
    if not note_hash and isinstance(existing_payload, Mapping):
        note_hash = coerce_text(
            existing_payload.get("note_hash"), preserve_case=True
        )

    fingerprint_hash = coerce_text(
        pack_payload.get("fingerprint_hash"), preserve_case=True
    )
    if not fingerprint_hash and isinstance(existing_payload, Mapping):
        fingerprint_hash = coerce_text(
            existing_payload.get("fingerprint_hash"), preserve_case=True
        )

    parsed = _parse_response_payload(response_payload)
    analysis_payload: Mapping[str, Any]
    if isinstance(parsed.get("analysis"), Mapping):
        analysis_payload = parsed["analysis"]  # type: ignore[assignment]
    else:
        analysis_payload = parsed

    normalized_analysis = validate_analysis_payload(analysis_payload)
    evaluated_at = _now_iso()

    result_payload: MutableMapping[str, Any] = {
        "sid": sid,
        "account_id": str(account_id),
        "prompt_salt": prompt_salt,
        "note_hash": note_hash,
        "analysis": normalized_analysis,
        "evaluated_at": evaluated_at,
        "fingerprint_hash": fingerprint_hash,
    }

    account_context_payload = pack_payload.get("account_context")
    if account_context_payload is not None:
        result_payload["account_context"] = account_context_payload

    bureaus_summary_payload = pack_payload.get("bureaus_summary")
    if bureaus_summary_payload is not None:
        result_payload["bureaus_summary"] = bureaus_summary_payload

    if isinstance(existing_payload, Mapping):
        note_metrics = existing_payload.get("note_metrics")
        if isinstance(note_metrics, Mapping):
            result_payload["note_metrics"] = dict(note_metrics)

    if "note_metrics" not in result_payload:
        metrics = pack_payload.get("note_metrics")
        if isinstance(metrics, Mapping):
            result_payload["note_metrics"] = dict(metrics)

    log.info(
        "STYLE_INGEST_RESULT sid=%s account_id=%s prompt_salt=%s note_hash=%s",
        sid,
        account_id,
        prompt_salt,
        note_hash,
    )

    return store_note_style_result(
        sid,
        account_id,
        result_payload,
        runs_root=runs_root,
        completed_at=evaluated_at,
    )


__all__ = ["ingest_note_style_result"]
