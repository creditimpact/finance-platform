"""Helpers for ingesting AI validation responses."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from backend.ai.validation_index import ValidationPackIndexWriter
from backend.core.ai.paths import (
    ensure_validation_paths,
    validation_pack_filename_for_account,
    validation_result_jsonl_filename_for_account,
    validation_result_summary_filename_for_account,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def _resolve_runs_root(runs_root: Path | str | None) -> Path:
    if runs_root is None:
        return Path("runs").resolve()
    return Path(runs_root).resolve()


def _index_writer(
    sid: str, runs_root: Path, paths: "ValidationPaths | None" = None
) -> ValidationPackIndexWriter:
    validation_paths = paths or ensure_validation_paths(runs_root, sid, create=True)
    return ValidationPackIndexWriter(
        sid=sid,
        index_path=validation_paths.index_file,
        packs_dir=validation_paths.packs_dir,
        results_dir=validation_paths.results_dir,
    )


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
    validation_paths = ensure_validation_paths(runs_root_path, sid, create=True)
    pack_filename = validation_pack_filename_for_account(account_id)
    pack_path = validation_paths.packs_dir / pack_filename
    writer = _index_writer(sid, runs_root_path, validation_paths)
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


def _coerce_account_int(account_id: int | str) -> int | None:
    try:
        return int(account_id)
    except (TypeError, ValueError):
        try:
            return int(str(account_id).strip())
        except (TypeError, ValueError):
            return None


def _load_pack_lookup(pack_path: Path) -> dict[str, Mapping[str, Any]]:
    lookup: dict[str, Mapping[str, Any]] = {}
    try:
        raw_lines = pack_path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return lookup
    except OSError:
        return lookup

    for line in raw_lines:
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, Mapping):
            continue

        for key_name in ("id", "field_key", "field"):
            key_value = payload.get(key_name)
            if isinstance(key_value, str) and key_value.strip():
                lookup.setdefault(key_value.strip(), payload)

    return lookup


def _normalize_decision(decision: Any) -> str:
    value = str(decision or "").strip().lower()
    if value == "strong":
        return "strong"
    if value in {"no_case", "no_claim", "no_claims"}:
        return "no_case"
    if value in {"", "unknown"}:
        return "no_case"
    return "no_case"


def _normalize_citations(raw: Any) -> list[str]:
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return []
    citations: list[str] = []
    for item in raw:
        if isinstance(item, str) and item.strip():
            citations.append(item.strip())
    return citations


def _fallback_result_id(account_int: int | None, field_name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", field_name.lower()).strip("_")
    if not slug:
        slug = "field"
    if account_int is None:
        return f"acc__{slug}"
    return f"acc_{account_int:03d}__{slug}"


def _collect_result_entries(payload: Mapping[str, Any]) -> Iterable[Mapping[str, Any]]:
    for key in ("results", "decision_per_field"):
        raw = payload.get(key)
        if isinstance(raw, Sequence):
            for entry in raw:
                if isinstance(entry, Mapping):
                    yield entry


def _build_result_lines(
    account_id: int | str,
    payload: Mapping[str, Any],
    pack_lookup: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    account_int = _coerce_account_int(account_id)
    result_lines: list[dict[str, Any]] = []

    for entry in _collect_result_entries(payload):
        candidate_keys: list[str] = []
        for key_name in ("id", "field_key", "field"):
            value = entry.get(key_name)
            if isinstance(value, str) and value.strip():
                candidate_keys.append(value.strip())

        pack_payload: Mapping[str, Any] | None = None
        for key in candidate_keys:
            pack_payload = pack_lookup.get(key)
            if pack_payload:
                break

        if pack_payload is not None:
            field_name = (
                str(pack_payload.get("field") or "").strip() or candidate_keys[-1]
            )
            line_id = str(pack_payload.get("id") or "").strip()
        else:
            field_name = candidate_keys[-1] if candidate_keys else ""
            line_id = ""

        if not field_name:
            continue

        if not line_id:
            line_id = _fallback_result_id(account_int, field_name)

        rationale = entry.get("rationale")
        if not isinstance(rationale, str):
            rationale = ""

        result_line = {
            "id": line_id,
            "account_id": account_int if account_int is not None else account_id,
            "field": field_name,
            "decision": _normalize_decision(entry.get("decision")),
            "rationale": rationale,
            "citations": _normalize_citations(entry.get("citations")),
        }

        result_lines.append(result_line)

    return result_lines


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
    validation_paths = ensure_validation_paths(runs_root_path, sid, create=True)
    summary_filename = validation_result_summary_filename_for_account(account_id)
    jsonl_filename = validation_result_jsonl_filename_for_account(account_id)
    summary_path = validation_paths.results_dir / summary_filename
    jsonl_path = validation_paths.results_dir / jsonl_filename

    pack_filename = validation_pack_filename_for_account(account_id)
    pack_path = validation_paths.packs_dir / pack_filename
    pack_lookup = _load_pack_lookup(pack_path)

    result_lines = _build_result_lines(account_id, response_payload, pack_lookup)

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

    normalized_payload["results"] = result_lines

    serialized_summary = json.dumps(
        normalized_payload, ensure_ascii=False, sort_keys=True
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(serialized_summary + "\n", encoding="utf-8")

    jsonl_lines = [
        json.dumps(line, ensure_ascii=False, sort_keys=True) for line in result_lines
    ]
    jsonl_contents = "\n".join(jsonl_lines)
    if jsonl_contents:
        jsonl_contents += "\n"
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    jsonl_path.write_text(jsonl_contents, encoding="utf-8")

    writer = _index_writer(sid, runs_root_path, validation_paths)
    writer.record_result(
        pack_path,
        status=normalized_status,
        error=error,
        request_lines=request_lines,
        model=normalized_payload.get("model"),
        completed_at=normalized_payload.get("completed_at"),
    )

    return summary_path


__all__ = ["mark_validation_pack_sent", "store_validation_result"]

