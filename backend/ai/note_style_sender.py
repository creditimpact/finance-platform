"""Execution helpers for the note_style AI stage."""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from backend.ai.note_style_ingest import ingest_note_style_result
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


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


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


def _atomic_write_index(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + f".tmp.{uuid.uuid4().hex}")
    try:
        with tmp_path.open("w", encoding="utf-8", newline="") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.flush()
            try:
                os.fsync(handle.fileno())
            except OSError:
                pass
        os.replace(tmp_path, path)
    finally:
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass
    try:
        fd = os.open(str(path.parent), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        try:
            os.close(fd)
        except OSError:
            pass


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

    container_key, indexed_entries, container = _extract_entries(document)
    if container_key is None:
        return []

    targets: list[tuple[str, str, int]] = []
    for idx, entry in indexed_entries:
        status = _normalize_status(entry.get("status"))
        if status != "built":
            continue
        account = str(entry.get("account_id") or "")
        pack_rel = str(entry.get("pack") or "")
        if not account or not pack_rel:
            continue
        targets.append((account, pack_rel, idx))

    if not targets:
        return []

    client = get_ai_client()
    processed: list[str] = []

    for account_id, pack_rel, idx in targets:
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

        sent_at = _now_iso()
        entry = dict(container[idx])
        entry["status"] = "completed"
        entry["sent_at"] = sent_at
        entry["result"] = _relativize(result_path, paths.base)
        container[idx] = entry
        _atomic_write_index(paths.index_file, document)

        log.info(
            "STYLE_SEND_ACCOUNT_END sid=%s account_id=%s status=completed", sid, account_id
        )
        processed.append(account_id)

    return processed


__all__ = ["send_note_style_packs_for_sid"]
