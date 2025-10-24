"""Execution helpers for the note_style AI stage."""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from backend.ai.note_style_ingest import ingest_note_style_result
from backend.ai.note_style_results import (
    record_note_style_failure,
    store_note_style_result,
)
from backend.ai.note_style_logging import log_structured_event
from backend.core.ai.paths import (
    NoteStyleAccountPaths,
    NoteStylePaths,
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


def _resolve_packs_dir(paths: NoteStylePaths) -> Path:
    override = os.getenv("NOTE_STYLE_PACKS_DIR")
    if override:
        candidate = Path(override)
        if not candidate.is_absolute():
            candidate = (paths.base / override).resolve()
        else:
            candidate = candidate.resolve()
        return candidate
    return paths.packs_dir


def _is_within_directory(path: Path, base: Path) -> bool:
    try:
        path.resolve().relative_to(base.resolve())
    except ValueError:
        return False
    return True


def _load_pack_records(pack_path: Path) -> list[Mapping[str, Any]]:
    try:
        raw = pack_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise FileNotFoundError(f"Pack file not found: {pack_path}") from None
    except OSError as exc:
        raise RuntimeError(f"Failed to read pack file: {pack_path}") from exc

    payloads: list[Mapping[str, Any]] = []
    for line in raw.splitlines():
        candidate = line.strip()
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in pack file: {pack_path}") from exc
        if not isinstance(parsed, Mapping):
            raise ValueError(f"Pack payload must be an object: {pack_path}")
        payloads.append(parsed)

    if not payloads:
        raise ValueError(f"Pack file is empty: {pack_path}")
    return payloads


def _load_pack_payload(pack_path: Path) -> Mapping[str, Any]:
    return _load_pack_records(pack_path)[0]


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


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _safe_json_payload(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _safe_json_payload(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_safe_json_payload(item) for item in value]
    if hasattr(value, "model_dump_json"):
        try:
            return json.loads(value.model_dump_json())
        except Exception:
            pass
    if hasattr(value, "dict") and callable(getattr(value, "dict")):
        try:
            return value.dict()  # type: ignore[call-arg]
        except Exception:
            pass
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _extract_response_text(response_payload: Any) -> str:
    choices: Sequence[Any] | None = None
    if hasattr(response_payload, "choices"):
        choices = getattr(response_payload, "choices")
    elif isinstance(response_payload, Mapping):
        choices = response_payload.get("choices")  # type: ignore[assignment]

    if not isinstance(choices, Sequence) or not choices:
        return ""

    first = choices[0]
    if hasattr(first, "message"):
        message = getattr(first, "message")
    elif isinstance(first, Mapping):
        message = first.get("message")
    else:
        message = None

    if message is None:
        return ""

    if hasattr(message, "content"):
        content = getattr(message, "content")
    elif isinstance(message, Mapping):
        content = message.get("content")
    else:
        content = None

    if isinstance(content, str):
        return content
    if isinstance(content, Sequence) and not isinstance(content, (bytes, bytearray)):
        pieces: list[str] = []
        for chunk in content:
            if isinstance(chunk, str):
                pieces.append(chunk)
            elif isinstance(chunk, Mapping):
                text_piece = chunk.get("text")
                if isinstance(text_piece, str):
                    pieces.append(text_piece)
        return "".join(pieces)
    if content is not None:
        return str(content)
    return ""


def _write_raw_response(
    *,
    account_paths: NoteStyleAccountPaths,
    response_payload: Any,
) -> Path:
    """Persist the raw model content for debugging."""

    base_path = account_paths.result_raw_file
    base_path.parent.mkdir(parents=True, exist_ok=True)

    text_content = _extract_response_text(response_payload)
    text_to_write = text_content if isinstance(text_content, str) else ""
    stripped = text_to_write.strip()

    target_path = base_path
    if stripped:
        try:
            json.loads(stripped)
        except json.JSONDecodeError:
            pass
        else:
            target_path = base_path.with_suffix(".json")
            if text_to_write and not text_to_write.endswith("\n"):
                text_to_write = f"{text_to_write}\n"
    if target_path == base_path:
        if stripped:
            if text_to_write and not text_to_write.endswith("\n"):
                text_to_write = f"{text_to_write}\n"
        else:
            safe_payload = _safe_json_payload(response_payload)
            target_path = base_path.with_suffix(".json")
            text_to_write = json.dumps(safe_payload, ensure_ascii=False, indent=2)
            text_to_write += "\n"

    with target_path.open("w", encoding="utf-8") as handle:
        handle.write(text_to_write)

    alternate = base_path.with_suffix(".json")
    for candidate in (base_path, alternate):
        if candidate != target_path and candidate.exists():
            try:
                candidate.unlink()
            except OSError:
                log.debug(
                    "STYLE_SEND_RAW_CLEANUP_FAILED path=%s", candidate, exc_info=True
                )

    return target_path


def _write_invalid_result_marker(
    *, account_paths: NoteStyleAccountPaths, reason: str
) -> Path:
    marker = {"error": "invalid_result", "reason": reason, "at": _now_iso()}
    result_path = account_paths.result_file
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with result_path.open("w", encoding="utf-8") as handle:
        json.dump(marker, handle, ensure_ascii=False)
        handle.write("\n")
    return result_path


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
    packs_dir = _resolve_packs_dir(paths)
    debug_dir = getattr(paths, "debug_dir", paths.base / "debug")
    glob_pattern = os.getenv("NOTE_STYLE_PACK_GLOB") or "style_*.jsonl"
    pack_candidates = sorted(packs_dir.glob(glob_pattern))

    if not pack_candidates:
        return []

    client = get_ai_client()
    processed: list[str] = []
    processed_accounts: set[str] = set()

    for pack_path in pack_candidates:
        if not _is_within_directory(pack_path, packs_dir):
            log.warning(
                "STYLE_SEND_PACK_OUTSIDE_DIR sid=%s path=%s packs_dir=%s",
                sid,
                pack_path,
                packs_dir,
            )
            continue
        if _is_within_directory(pack_path, debug_dir):
            log.info(
                "STYLE_SEND_SKIP_DEBUG sid=%s path=%s",
                sid,
                pack_path,
            )
            continue
        if not pack_path.is_file():
            continue
        try:
            pack_records = _load_pack_records(pack_path)
        except Exception:
            log.exception(
                "STYLE_SEND_PACK_LOAD_FAILED sid=%s path=%s", sid, pack_path
            )
            raise

        for pack_payload in pack_records:
            account_id = str(pack_payload.get("account_id") or "").strip()
            if not account_id:
                continue
            if account_id in processed_accounts:
                continue
            processed_accounts.add(account_id)

            account_paths = ensure_note_style_account_paths(
                paths, account_id, create=True
            )

            log.info(
                "STYLE_SEND_ACCOUNT_START sid=%s account_id=%s pack=%s",
                sid,
                account_id,
                pack_path,
            )

            model = str(pack_payload.get("model") or "")
            messages = _coerce_messages(pack_payload)

            start = time.perf_counter()
            try:
                response = client.chat_completion(
                    model=model or None,
                    messages=list(messages),
                    temperature=0,
                    response_format={"type": "json_object"},
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
            except ValueError as exc:
                log.warning(
                    "STYLE_SEND_RESULTS_FAILED sid=%s account_id=%s reason=parse_error",
                    sid,
                    account_id,
                    exc_info=True,
                )
                raw_path = _write_raw_response(
                    account_paths=account_paths,
                    response_payload=response,
                )
                reason_text = str(exc).strip() or exc.__class__.__name__
                marker_path = _write_invalid_result_marker(
                    account_paths=account_paths, reason=reason_text
                )
                log.warning(
                    "[NOTE_STYLE] RESULT_INVALID account=%s reason=%s raw_path=%s result_path=%s",
                    account_id,
                    reason_text,
                    raw_path.resolve().as_posix(),
                    marker_path.resolve().as_posix(),
                )
                record_note_style_failure(
                    sid,
                    account_id,
                    runs_root=runs_root_path,
                    error=str(exc),
                )
                continue
            except NotImplementedError:
                written_path = account_paths.result_file
            except Exception as exc:
                log.exception(
                    "STYLE_SEND_RESULTS_FAILED sid=%s account_id=%s",
                    sid,
                    account_id,
                )
                _write_raw_response(
                    account_paths=account_paths,
                    response_payload=response,
                )
                record_note_style_failure(
                    sid,
                    account_id,
                    runs_root=runs_root_path,
                    error=str(exc),
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
                "STYLE_SEND_ACCOUNT_END sid=%s account_id=%s status=completed",
                sid,
                account_id,
            )
            processed.append(account_id)

    return processed


__all__ = ["send_note_style_packs_for_sid"]
