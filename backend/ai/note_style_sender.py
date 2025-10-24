"""Execution helpers for the note_style AI stage."""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from backend.ai.note_style_ingest import ingest_note_style_result
from backend.ai.note_style_results import store_note_style_result
from backend.ai.note_style_logging import log_structured_event
from backend.core.ai.paths import (
    ensure_note_style_account_paths,
    ensure_note_style_paths,
)
from backend.core.services.ai_client import get_ai_client


log = logging.getLogger(__name__)


def _resolve_runs_root(runs_root: Path | str | None) -> Path:
    if runs_root is None:
        env_value = os.getenv("RUNS_ROOT")
        return Path(env_value) if env_value else Path("runs")
    return Path(runs_root)


def _load_json_mapping(path: Path) -> Mapping[str, Any] | None:
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError:
        log.warning("STYLE_SEND_INDEX_READ_FAILED path=%s", path, exc_info=True)
        return None

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        log.warning("STYLE_SEND_INDEX_PARSE_FAILED path=%s", path, exc_info=True)
        return None

    if isinstance(payload, Mapping):
        return payload
    return None


def _extract_entries(
    document: Mapping[str, Any]
) -> tuple[str | None, list[tuple[int, Mapping[str, Any]]], list[Any]]:
    for key in ("packs", "items"):
        container = document.get(key)
        if isinstance(container, list):
            extracted: list[tuple[int, Mapping[str, Any]]] = []
            for idx, entry in enumerate(container):
                if isinstance(entry, Mapping):
                    extracted.append((idx, entry))
            return key, extracted, container
    return None, [], []


def _normalize_status(value: Any) -> str:
    if isinstance(value, str):
        return value.strip().lower()
    if isinstance(value, (bytes, bytearray)):
        try:
            return bytes(value).decode("utf-8", errors="ignore").strip().lower()
        except Exception:  # pragma: no cover - defensive
            return ""
    return ""


def _load_pack_payload(pack_path: Path) -> Mapping[str, Any]:
    try:
        raw = pack_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise FileNotFoundError(f"Pack file not found: {pack_path}") from None
    except OSError as exc:
        raise RuntimeError(f"Failed to read pack file: {pack_path}") from exc

    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"Pack file is empty: {pack_path}")

    try:
        payload = json.loads(lines[0])
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in pack file: {pack_path}") from exc

    if not isinstance(payload, Mapping):
        raise ValueError(f"Pack payload must be an object: {pack_path}")

    return payload


def _coerce_messages(payload: Mapping[str, Any]) -> Sequence[Mapping[str, Any]]:
    messages = payload.get("messages")
    if not isinstance(messages, Sequence):
        raise ValueError("Pack payload missing messages sequence")
    normalized: list[Mapping[str, Any]] = []
    for entry in messages:
        if not isinstance(entry, Mapping):
            raise ValueError("Pack messages must be mapping objects")
        normalized.append(entry)
    return normalized


def _relativize(path: Path, base: Path) -> str:
    resolved_path = path.resolve()
    resolved_base = base.resolve()
    try:
        relative = resolved_path.relative_to(resolved_base)
    except ValueError:
        relative = Path(os.path.relpath(resolved_path, resolved_base))
    return str(PurePosixPath(relative))


def _coerce_result_path(value: Any) -> Path | None:
    if value is None:
        return None
    if isinstance(value, Path):
        return value
    try:
        text = os.fspath(value)
    except TypeError:
        text = str(value)
    try:
        return Path(text)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return None


def _load_existing_result(path: Path) -> Mapping[str, Any] | None:
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError:
        log.warning("STYLE_SEND_RESULT_READ_FAILED path=%s", path, exc_info=True)
        return None

    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    if not lines:
        return None

    try:
        payload = json.loads(lines[0])
    except json.JSONDecodeError:
        log.warning("STYLE_SEND_RESULT_INVALID_JSON path=%s", path, exc_info=True)
        return None

    if isinstance(payload, Mapping):
        return payload
    return None


def send_note_style_packs_for_sid(
    sid: str,
    *,
    runs_root: Path | str | None = None,
) -> list[str]:
    """Send all note_style packs with ``status="built"`` for ``sid``.

    Returns a list of account identifiers that were processed. Raises any
    exception encountered while reading packs, invoking the model, or persisting
    results so the caller can handle retries.
    """

    runs_root_path = _resolve_runs_root(runs_root)
    paths = ensure_note_style_paths(runs_root_path, sid, create=False)
    document = _load_json_mapping(paths.index_file)
    if not isinstance(document, Mapping):
        return []

    container_key, indexed_entries, _ = _extract_entries(document)
    if container_key is None:
        return []

    targets: list[tuple[str, str]] = []
    for idx, entry in indexed_entries:
        status = _normalize_status(entry.get("status"))
        if status != "built":
            continue
        account = str(entry.get("account_id") or "")
        pack_rel = str(entry.get("pack") or "")
        if not account or not pack_rel:
            continue
        targets.append((account, pack_rel))

    if not targets:
        return []

    client = get_ai_client()
    processed: list[str] = []

    for account_id, pack_rel in targets:
        account_paths = ensure_note_style_account_paths(paths, account_id, create=True)
        pack_path = Path(pack_rel)
        if not pack_path.is_absolute():
            pack_path = (paths.base / pack_path).resolve()

        log.info(
            "STYLE_SEND_ACCOUNT_START sid=%s account_id=%s pack=%s", sid, account_id, pack_path
        )

        pack_payload = _load_pack_payload(pack_path)
        model = str(pack_payload.get("model") or "")
        messages = _coerce_messages(pack_payload)
        pack_note_hash = str(pack_payload.get("note_hash") or "")

        existing_result = _load_existing_result(account_paths.result_file)
        existing_note_hash = ""
        analysis_ready = False
        evaluated_at: str | None = None
        if isinstance(existing_result, Mapping):
            existing_note_hash = str(existing_result.get("note_hash") or "")
            evaluated_at_value = existing_result.get("evaluated_at")
            if isinstance(evaluated_at_value, str) and evaluated_at_value:
                evaluated_at = evaluated_at_value
            analysis_payload = existing_result.get("analysis")
            if isinstance(analysis_payload, Mapping) and analysis_payload:
                analysis_ready = True

        if analysis_ready and existing_note_hash and existing_note_hash == pack_note_hash:
            log.info(
                "STYLE_SEND_SKIPPED sid=%s account_id=%s reason=existing_result", sid, account_id
            )
            store_note_style_result(
                sid=sid,
                account_id=account_id,
                payload=dict(existing_result),  # type: ignore[arg-type]
                runs_root=runs_root_path,
                completed_at=evaluated_at,
            )
            processed.append(account_id)
            continue

        start = time.perf_counter()
        try:
            response = client.chat_completion(
                model=model or None,
                messages=list(messages),
                temperature=0,
            )
            latency = time.perf_counter() - start
            log.info(
                "STYLE_SEND_MODEL_CALL sid=%s account_id=%s model=%s status=success latency=%.3fs",
                sid,
                account_id,
                model or "",
                latency,
            )
        except Exception:
            latency = time.perf_counter() - start
            log.exception(
                "STYLE_SEND_MODEL_CALL sid=%s account_id=%s model=%s status=error latency=%.3fs",
                sid,
                account_id,
                model or "",
                latency,
            )
            raise

        try:
            written_path = ingest_note_style_result(
                sid=sid,
                account_id=account_id,
                runs_root=runs_root_path,
                account_paths=account_paths,
                pack_payload=pack_payload,
                response_payload=response,
            )
        except NotImplementedError:
            written_path = account_paths.result_file
        except Exception:
            log.exception(
                "STYLE_SEND_RESULTS_FAILED sid=%s account_id=%s", sid, account_id
            )
            raise

        result_path = _coerce_result_path(written_path) or account_paths.result_file
        log.info(
            "STYLE_SEND_RESULTS_WRITTEN sid=%s account_id=%s result=%s",
            sid,
            account_id,
            result_path,
        )

        result_relative = _relativize(result_path, paths.base)
        pack_relative = _relativize(pack_path, paths.base)
        log_structured_event(
            "NOTE_STYLE_SENT_OK",
            logger=log,
            sid=sid,
            account_id=account_id,
            model=model or "",
            latency_seconds=latency,
            pack_path=pack_relative,
            result_path=result_relative,
        )

        log.info(
            "STYLE_SEND_ACCOUNT_END sid=%s account_id=%s status=completed", sid, account_id
        )
        processed.append(account_id)

    return processed


__all__ = ["send_note_style_packs_for_sid"]
