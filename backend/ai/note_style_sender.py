"""Execution helpers for the note_style AI stage."""

from __future__ import annotations

import json
import logging
import os
import time
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from backend import config
from backend.ai.note_style_ingest import ingest_note_style_result
from backend.ai.note_style_results import record_note_style_failure
from backend.ai.note_style_logging import log_structured_event
from backend.core.ai.paths import (
    NoteStyleAccountPaths,
    NoteStylePaths,
    ensure_note_style_account_paths,
    normalize_note_style_account_id,
)
from backend.core.paths import normalize_worker_path
from backend.core.services.ai_client import get_ai_client
from backend.runflow.manifest import resolve_note_style_stage_paths


log = logging.getLogger(__name__)


_INDEX_THIN_THRESHOLD_BYTES = 128


_PATH_LOG_CACHE: set[str] = set()


def _log_sender_paths(sid: str, paths: NoteStylePaths) -> None:
    signature = "|".join(
        [
            sid,
            paths.base.as_posix(),
            paths.packs_dir.as_posix(),
            paths.results_dir.as_posix(),
            paths.index_file.as_posix(),
            paths.log_file.as_posix(),
        ]
    )

    if signature in _PATH_LOG_CACHE:
        return

    _PATH_LOG_CACHE.add(signature)
    log.info(
        "NOTE_STYLE_SEND_PATHS sid=%s base=%s packs=%s results=%s index=%s logs=%s manifest_paths=%s",
        sid,
        paths.base,
        paths.packs_dir,
        paths.results_dir,
        paths.index_file,
        paths.log_file,
        config.NOTE_STYLE_USE_MANIFEST_PATHS,
    )


def _resolve_runs_root(runs_root: Path | str | None) -> Path:
    def _coerce(value: Path | str) -> Path:
        if isinstance(value, Path):
            return value.resolve()

        text = str(value or "").strip()
        if not text:
            return Path("runs").resolve()

        sanitized = text.replace("\\", "/")
        try:
            return normalize_worker_path(Path.cwd(), sanitized)
        except ValueError:
            return Path("runs").resolve()

    if runs_root is None:
        env_value = os.getenv("RUNS_ROOT")
        if env_value:
            return _coerce(env_value)
        return Path("runs").resolve()

    return _coerce(runs_root)


def _resolve_packs_dir(paths: NoteStylePaths) -> Path:
    override = os.getenv("NOTE_STYLE_PACKS_DIR")
    if override:
        run_dir = paths.base.parent.parent
        try:
            candidate = normalize_worker_path(run_dir, override)
        except ValueError:
            candidate = paths.packs_dir
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


_DISALLOWED_MESSAGE_KEY_SUBSTRINGS: tuple[str, ...] = (
    "debug",
    "snapshot",
    "raw",
    "blob",
)


def _sanitize_message_entry(entry: Mapping[str, Any]) -> Mapping[str, Any]:
    sanitized: dict[str, Any] = {}
    for key, value in entry.items():
        key_text = str(key)
        lower_key = key_text.lower()
        if any(fragment in lower_key for fragment in _DISALLOWED_MESSAGE_KEY_SUBSTRINGS):
            continue
        sanitized[key_text] = value

    if "role" not in sanitized or "content" not in sanitized:
        raise ValueError("Pack message missing required role/content fields")

    return sanitized


def _coerce_messages(payload: Mapping[str, Any]) -> Sequence[Mapping[str, Any]]:
    messages = payload.get("messages")
    if not isinstance(messages, Sequence):
        raise ValueError("Pack payload missing messages sequence")
    normalized: list[Mapping[str, Any]] = []
    for entry in messages:
        if not isinstance(entry, Mapping):
            raise ValueError("Pack messages must be mapping objects")
        normalized.append(_sanitize_message_entry(entry))
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


def _account_id_from_pack_path(pack_path: Path) -> str:
    stem = pack_path.stem
    if stem.startswith("acc_"):
        return stem[4:]
    return stem


def _load_index_account_map(paths: NoteStylePaths) -> dict[str, str]:
    index_path = getattr(paths, "index_file", None)
    if not isinstance(index_path, Path):
        return {}

    try:
        raw = index_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    except OSError:
        log.warning(
            "STYLE_SEND_INDEX_READ_FAILED path=%s", index_path, exc_info=True
        )
        return {}

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        log.warning("STYLE_SEND_INDEX_INVALID_JSON path=%s", index_path)
        return {}

    mapping: dict[str, str] = {}
    if isinstance(payload, Mapping):
        entries = payload.get("packs")
        if isinstance(entries, Sequence):
            for entry in entries:
                if not isinstance(entry, Mapping):
                    continue
                account_id = str(entry.get("account_id") or "").strip()
                pack_path_value = entry.get("pack_path")
                if not account_id:
                    continue
                if isinstance(pack_path_value, str):
                    normalized = pack_path_value.strip()
                    if normalized:
                        mapping[normalized] = account_id
    return mapping


@dataclass(frozen=True)
class _PackCandidate:
    pack_path: Path
    result_path: Path | None = None
    account_id: str | None = None
    normalized_account_id: str | None = None


def _normalize_manifest_entry_path(
    value: Any,
    *,
    paths: NoteStylePaths,
    default_dir: Path,
) -> Path | None:
    if value is None:
        return None

    try:
        text = os.fspath(value)
    except TypeError:
        text = str(value)

    sanitized = str(text).strip()
    if not sanitized:
        return None

    sanitized = sanitized.replace("\\", "/")
    run_dir = paths.base.parent.parent

    try:
        candidate = normalize_worker_path(run_dir, sanitized)
    except ValueError:
        return None

    if not _is_within_directory(candidate, run_dir):
        return None

    stage_base = paths.base.resolve()
    stage_name = stage_base.name.lower()
    sanitized_relative = sanitized.lstrip("./")
    if stage_name and sanitized_relative.lower().startswith(f"{stage_name}/"):
        sanitized_for_stage = sanitized_relative.split("/", 1)[1]
    else:
        sanitized_for_stage = sanitized

    if not _is_within_directory(candidate, stage_base):
        try:
            candidate = normalize_worker_path(
                default_dir.parent, sanitized_for_stage
            )
        except ValueError:
            candidate = default_dir / Path(sanitized_for_stage)

        if not _is_within_directory(candidate, run_dir):
            return None

    return candidate.resolve()


def _load_manifest_pack_entries(
    paths: NoteStylePaths, *, sid: str | None = None
) -> list[_PackCandidate]:
    index_path = paths.index_file
    try:
        raw = index_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return []
    except OSError:
        log.warning(
            "STYLE_SEND_MANIFEST_INDEX_READ_FAILED sid=%s path=%s",
            sid,
            index_path,
            exc_info=True,
        )
        return []

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        log.warning(
            "STYLE_SEND_MANIFEST_INDEX_INVALID_JSON sid=%s path=%s",
            sid,
            index_path,
            exc_info=True,
        )
        return []

    pack_entries = payload.get("packs")
    if not isinstance(pack_entries, Sequence):
        return []

    candidates: list[_PackCandidate] = []
    for entry in pack_entries:
        if not isinstance(entry, Mapping):
            continue

        pack_path_value = entry.get("pack_path") or entry.get("pack")
        pack_path = _normalize_manifest_entry_path(
            pack_path_value,
            paths=paths,
            default_dir=paths.packs_dir,
        )
        if pack_path is None:
            log.warning(
                "STYLE_SEND_MANIFEST_PACK_INVALID sid=%s value=%s",
                sid,
                pack_path_value,
            )
            continue

        result_path_value = entry.get("result_path") or entry.get("result")
        result_path = _normalize_manifest_entry_path(
            result_path_value,
            paths=paths,
            default_dir=paths.results_dir,
        )

        account_raw = entry.get("account_id")
        account_id: str | None
        if account_raw is None:
            account_id = None
        else:
            account_text = str(account_raw).strip()
            account_id = account_text or None

        normalized = (
            normalize_note_style_account_id(account_id)
            if account_id is not None
            else None
        )

        candidates.append(
            _PackCandidate(
                pack_path=pack_path,
                result_path=result_path,
                account_id=account_id,
                normalized_account_id=normalized,
            )
        )

    return candidates


def _warn_if_index_thin(paths: NoteStylePaths, *, sid: str) -> None:
    if not config.NOTE_STYLE_WAIT_FOR_INDEX:
        return

    index_path = getattr(paths, "index_file", None)
    if not isinstance(index_path, Path):
        return

    display_path = _relativize(index_path, paths.base)

    try:
        size = index_path.stat().st_size
    except FileNotFoundError:
        log.warning(
            "NOTE_STYLE_INDEX_THIN sid=%s path=%s reason=missing",
            sid,
            display_path,
        )
        return
    except OSError:
        log.warning(
            "NOTE_STYLE_INDEX_THIN sid=%s path=%s reason=stat_failed",
            sid,
            display_path,
            exc_info=True,
        )
        return

    if size < _INDEX_THIN_THRESHOLD_BYTES:
        log.warning(
            "NOTE_STYLE_INDEX_THIN sid=%s bytes=%s threshold=%s path=%s",
            sid,
            size,
            _INDEX_THIN_THRESHOLD_BYTES,
            display_path,
        )


def _account_paths_for_candidate(
    paths: NoteStylePaths,
    account_id: str,
    candidate: _PackCandidate | None,
) -> NoteStyleAccountPaths:
    account_paths = ensure_note_style_account_paths(paths, account_id, create=True)

    if candidate is None:
        return account_paths

    pack_file = candidate.pack_path
    result_file = candidate.result_path or account_paths.result_file

    if candidate.result_path is not None:
        candidate.result_path.parent.mkdir(parents=True, exist_ok=True)

    return NoteStyleAccountPaths(
        account_id=account_paths.account_id,
        pack_file=pack_file,
        result_file=result_file,
        result_raw_file=account_paths.result_raw_file,
        debug_file=account_paths.debug_file,
    )


def _result_has_completed_analysis(result_path: Path) -> bool:
    try:
        raw = result_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return False
    except OSError:
        log.warning(
            "STYLE_SEND_EXISTING_READ_FAILED path=%s",
            result_path,
            exc_info=True,
        )
        return False

    for line in raw.splitlines():
        candidate = line.strip()
        if not candidate:
            continue
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            log.warning(
                "STYLE_SEND_EXISTING_INVALID_JSON path=%s",
                result_path,
                exc_info=True,
            )
            return False
        analysis = payload.get("analysis")
        if isinstance(analysis, Mapping) and bool(analysis):
            return True
        return False
    return False


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


def _extract_response_json(response_payload: Any) -> Mapping[str, Any]:
    text = _extract_response_text(response_payload)
    if not text or not text.strip():
        raise ValueError("Model response missing JSON content")

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("Model response is not valid JSON") from exc

    if not isinstance(parsed, Mapping):
        raise ValueError("Model response payload must be an object")

    return parsed


def _validate_response_structure(payload: Mapping[str, Any]) -> None:
    required_keys = {"tone", "context_hints", "emphasis", "confidence", "risk_flags"}
    missing = sorted(required_keys.difference(payload.keys()))
    if missing:
        raise ValueError(f"Model response missing required fields: {', '.join(missing)}")

    confidence = payload.get("confidence")
    try:
        numeric_confidence = float(confidence)
    except (TypeError, ValueError) as exc:
        raise ValueError("confidence must be a number between 0 and 1") from exc
    if math.isnan(numeric_confidence) or math.isinf(numeric_confidence):
        raise ValueError("confidence must be a finite number between 0 and 1")
    if numeric_confidence < 0 or numeric_confidence > 1:
        raise ValueError("confidence must be between 0 and 1")

    emphasis = payload.get("emphasis")
    if not isinstance(emphasis, Sequence) or isinstance(emphasis, (str, bytes, bytearray)):
        raise ValueError("emphasis must be an array")
    if len(emphasis) > 6:
        raise ValueError("emphasis must contain at most 6 items")

    risk_flags = payload.get("risk_flags")
    if not isinstance(risk_flags, Sequence) or isinstance(risk_flags, (str, bytes, bytearray)):
        raise ValueError("risk_flags must be an array")
    if len(risk_flags) > 6:
        raise ValueError("risk_flags must contain at most 6 items")


def _ensure_valid_json_response(response_payload: Any) -> None:
    payload = _extract_response_json(response_payload)
    _validate_response_structure(payload)


def _write_raw_response(
    *,
    account_paths: NoteStyleAccountPaths,
    response_payload: Any,
) -> Path:
    """Persist the raw model content for debugging."""

    raw_path = account_paths.result_raw_file
    raw_path.parent.mkdir(parents=True, exist_ok=True)

    text_content = _extract_response_text(response_payload)
    if isinstance(text_content, str) and text_content.strip():
        to_write = text_content
    else:
        safe_payload = _safe_json_payload(response_payload)
        to_write = json.dumps(safe_payload, ensure_ascii=False, indent=2)
    if not to_write.endswith("\n"):
        to_write = f"{to_write}\n"

    raw_path.write_text(to_write, encoding="utf-8")
    return raw_path


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


def _handle_invalid_response(
    *,
    sid: str,
    account_id: str,
    account_paths: NoteStyleAccountPaths,
    response_payload: Any,
    runs_root_path: Path,
    reason: str,
) -> None:
    raw_path = _write_raw_response(
        account_paths=account_paths,
        response_payload=response_payload,
    )
    marker_path = _write_invalid_result_marker(
        account_paths=account_paths, reason=reason
    )
    log.warning(
        "NOTE_STYLE_RESULT_INVALID sid=%s account_id=%s reason=%s raw_path=%s result_path=%s",
        sid,
        account_id,
        reason,
        raw_path.resolve().as_posix(),
        marker_path.resolve().as_posix(),
    )
    record_note_style_failure(
        sid,
        account_id,
        runs_root=runs_root_path,
        error=reason,
    )


def _send_pack_payload(
    *,
    sid: str,
    account_id: str,
    pack_payload: Mapping[str, Any],
    pack_relative: str,
    pack_path: Path,
    account_paths: NoteStyleAccountPaths,
    paths: NoteStylePaths,
    runs_root_path: Path,
    client: Any,
) -> bool:
    log.info(
        "STYLE_SEND_ACCOUNT_START sid=%s account_id=%s pack=%s",
        sid,
        account_id,
        pack_path,
    )

    if config.NOTE_STYLE_SKIP_IF_RESULT_EXISTS and _result_has_completed_analysis(
        account_paths.result_file
    ):
        log.info(
            "STYLE_SEND_SKIP_EXISTING sid=%s account_id=%s result=%s",
            sid,
            account_id,
            _relativize(account_paths.result_file, paths.base),
        )
        log_structured_event(
            "NOTE_STYLE_SEND_SKIPPED",
            logger=log,
            sid=sid,
            account_id=account_id,
            reason="existing_analysis",
            result_path=_relativize(account_paths.result_file, paths.base),
        )
        return False

    model = str(pack_payload.get("model") or "").strip()
    if not model:
        model = config.NOTE_STYLE_MODEL
    messages = _coerce_messages(pack_payload)

    start = time.perf_counter()
    try:
        response = client.chat_completion(
            model=model or None,
            messages=list(messages),
            temperature=0,
            response_format="json_object",
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
        _ensure_valid_json_response(response)
    except ValueError as exc:
        reason_text = str(exc).strip() or exc.__class__.__name__
        _handle_invalid_response(
            sid=sid,
            account_id=account_id,
            account_paths=account_paths,
            response_payload=response,
            runs_root_path=runs_root_path,
            reason=reason_text,
        )
        return False

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
        reason_text = str(exc).strip() or exc.__class__.__name__
        _handle_invalid_response(
            sid=sid,
            account_id=account_id,
            account_paths=account_paths,
            response_payload=response,
            runs_root_path=runs_root_path,
            reason=reason_text,
        )
        return False
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
        "NOTE_STYLE_SENT sid=%s account_id=%s result=%s",
        sid,
        account_id,
        result_path,
    )

    result_relative = _relativize(result_path, paths.base)
    log_structured_event(
        "NOTE_STYLE_SENT",
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
    return True


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
    paths = resolve_note_style_stage_paths(runs_root_path, sid, create=False)
    _log_sender_paths(sid, paths)
    _warn_if_index_thin(paths, sid=sid)
    packs_dir = _resolve_packs_dir(paths)
    debug_dir = getattr(paths, "debug_dir", paths.base / "debug")
    env_glob_raw = os.getenv("NOTE_STYLE_PACK_GLOB")
    env_glob = None
    if env_glob_raw:
        sanitized_glob = env_glob_raw.strip().replace("\\", "/")
        if sanitized_glob:
            env_glob = sanitized_glob
    fallback_glob = "acc_*.jsonl"
    default_glob = env_glob or fallback_glob

    log.info(
        "NOTE_STYLE_SEND_START sid=%s packs_dir=%s glob=%s use_manifest=%s",
        sid,
        packs_dir,
        "<manifest>" if config.NOTE_STYLE_USE_MANIFEST_PATHS else default_glob,
        config.NOTE_STYLE_USE_MANIFEST_PATHS,
    )

    manifest_candidates: list[_PackCandidate] = []
    if config.NOTE_STYLE_USE_MANIFEST_PATHS:
        manifest_candidates = _load_manifest_pack_entries(paths, sid=sid)

    pack_candidates: list[_PackCandidate]
    if manifest_candidates:
        pack_candidates = manifest_candidates
        log.info(
            "NOTE_STYLE_PACK_DISCOVERY sid=%s glob=%s matches=%s manifest=%s",
            sid,
            "<manifest>",
            len(pack_candidates),
            True,
        )
    else:
        glob_pattern = default_glob

        def _collect_candidates(pattern: str) -> list[Path]:
            try:
                raw_matches = sorted(packs_dir.glob(pattern))
            except ValueError:
                log.warning(
                    "NOTE_STYLE_PACK_GLOB_INVALID sid=%s glob=%s", sid, pattern
                )
                return []
            filtered: list[Path] = []
            packs_dir_resolved = packs_dir.resolve()
            debug_dir_resolved = debug_dir.resolve()
            for match in raw_matches:
                if match.is_dir():
                    continue
                if _is_within_directory(match, debug_dir_resolved):
                    continue
                try:
                    relative_parts = match.resolve().relative_to(packs_dir_resolved).parts
                except ValueError:
                    relative_parts = match.parts
                if any(part.startswith("results") for part in relative_parts):
                    continue
                if match.name.startswith("results"):
                    continue
                filtered.append(match)
            return filtered

        file_candidates = _collect_candidates(glob_pattern)
        effective_glob = glob_pattern

        if env_glob and not file_candidates and glob_pattern != fallback_glob:
            fallback_candidates = _collect_candidates(fallback_glob)
            if fallback_candidates:
                log.info(
                    "NOTE_STYLE_PACK_GLOB_FALLBACK sid=%s glob=%s fallback=%s matches=%s",
                    sid,
                    glob_pattern,
                    fallback_glob,
                    len(fallback_candidates),
                )
                file_candidates = fallback_candidates
                effective_glob = fallback_glob

        pack_candidates = [_PackCandidate(pack_path=path) for path in file_candidates]
        log.info(
            "NOTE_STYLE_PACK_DISCOVERY sid=%s glob=%s matches=%s",
            sid,
            effective_glob,
            len(pack_candidates),
        )

    sample_candidates = [
        _relativize(candidate.pack_path, paths.base)
        for candidate in pack_candidates[:5]
    ]
    log.info(
        "NOTE_STYLE_PACKS_FOUND sid=%s count=%s sample=%s",
        sid,
        len(pack_candidates),
        sample_candidates,
    )

    if not pack_candidates:
        log.info("NOTE_STYLE_NO_PACKS sid=%s", sid)
        return []

    client = get_ai_client()
    processed: list[str] = []
    index_account_map = _load_index_account_map(paths)

    for candidate in pack_candidates:
        pack_path = candidate.pack_path

        if manifest_candidates:
            run_dir = paths.base.parent.parent
            if not _is_within_directory(pack_path, run_dir):
                log.warning(
                    "STYLE_SEND_PACK_OUTSIDE_MANIFEST sid=%s path=%s run_dir=%s",
                    sid,
                    pack_path,
                    run_dir,
                )
                continue
        else:
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

        pack_relative = _relativize(pack_path, paths.base)

        for pack_payload in pack_records:
            account_id = str(pack_payload.get("account_id") or "").strip()
            if not account_id and candidate.account_id:
                account_id = candidate.account_id
            if not account_id:
                account_id = index_account_map.get(pack_relative, "")
            if not account_id and candidate.normalized_account_id:
                account_id = candidate.normalized_account_id
            if not account_id:
                account_id = _account_id_from_pack_path(pack_path)
            if not account_id:
                log.warning(
                    "STYLE_SEND_ACCOUNT_UNKNOWN sid=%s pack=%s", sid, pack_path
                )
                continue
            account_paths = _account_paths_for_candidate(paths, account_id, candidate)

            log.info(
                "NOTE_STYLE_SENDING sid=%s account_id=%s file=%s",
                sid,
                account_id,
                pack_relative,
            )

            if _send_pack_payload(
                sid=sid,
                account_id=account_id,
                pack_payload=pack_payload,
                pack_relative=pack_relative,
                pack_path=pack_path,
                account_paths=account_paths,
                paths=paths,
                runs_root_path=runs_root_path,
                client=client,
            ):
                processed.append(account_id)

    log.info(
        "NOTE_STYLE_SEND_DONE sid=%s processed=%s",
        sid,
        processed,
    )

    return processed


def send_note_style_pack_for_account(
    sid: str,
    account_id: str,
    *,
    runs_root: Path | str | None = None,
) -> bool:
    runs_root_path = _resolve_runs_root(runs_root)
    paths = resolve_note_style_stage_paths(runs_root_path, sid, create=False)
    _log_sender_paths(sid, paths)
    _warn_if_index_thin(paths, sid=sid)
    candidate: _PackCandidate | None = None
    if config.NOTE_STYLE_USE_MANIFEST_PATHS:
        target = normalize_note_style_account_id(account_id)
        for entry in _load_manifest_pack_entries(paths, sid=sid):
            normalized_entry = entry.normalized_account_id
            if normalized_entry is None and entry.account_id is not None:
                normalized_entry = normalize_note_style_account_id(entry.account_id)
            if normalized_entry == target:
                candidate = entry
                break

    account_paths = _account_paths_for_candidate(paths, account_id, candidate)

    pack_path = account_paths.pack_file
    if not pack_path.exists():
        log.info(
            "STYLE_SEND_PACK_MISSING sid=%s account_id=%s path=%s",
            sid,
            account_id,
            pack_path,
        )
        return False

    try:
        pack_records = _load_pack_records(pack_path)
    except Exception:
        log.exception(
            "STYLE_SEND_PACK_LOAD_FAILED sid=%s path=%s", sid, pack_path
        )
        raise

    pack_relative = _relativize(pack_path, paths.base)
    client = get_ai_client()

    for pack_payload in pack_records:
        if _send_pack_payload(
            sid=sid,
            account_id=account_id,
            pack_payload=pack_payload,
            pack_relative=pack_relative,
            pack_path=pack_path,
            account_paths=account_paths,
            paths=paths,
            runs_root_path=runs_root_path,
            client=client,
        ):
            return True

    return False


__all__ = ["send_note_style_packs_for_sid", "send_note_style_pack_for_account"]
