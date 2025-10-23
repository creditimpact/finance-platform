"""Helpers for persisting note_style model results."""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence

from backend.ai.manifest import ensure_note_style_section
from backend.core.ai.paths import (
    NoteStylePaths,
    ensure_note_style_account_paths,
    ensure_note_style_paths,
)
from backend.core.runflow import runflow_barriers_refresh
from backend.runflow.decider import (
    reconcile_umbrella_barriers,
    refresh_note_style_stage_from_index,
)

log = logging.getLogger(__name__)


def _resolve_runs_root(runs_root: Path | str | None) -> Path:
    if runs_root is None:
        env_value = os.getenv("RUNS_ROOT")
        return Path(env_value) if env_value else Path("runs")
    return Path(runs_root)


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _relative_to_base(path: Path, base: Path) -> str:
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _structured_repr(payload: Any) -> str:
    try:
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)
    except Exception:
        return str(payload)


def _fsync_directory(directory: Path) -> None:
    try:
        fd = os.open(str(directory), os.O_RDONLY)
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


def _atomic_write_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    serialized = json.dumps(payload, ensure_ascii=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + f".tmp.{uuid.uuid4().hex}")
    try:
        with tmp_path.open("w", encoding="utf-8", newline="") as handle:
            handle.write(serialized)
            handle.write("\n")
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
    _fsync_directory(path.parent)


def _relativize(path: Path, base: Path) -> str:
    resolved_path = path.resolve()
    resolved_base = base.resolve()
    try:
        relative = resolved_path.relative_to(resolved_base)
    except ValueError:
        relative = Path(os.path.relpath(resolved_path, resolved_base))
    return relative.as_posix()


def _compute_totals(entries: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    total = 0
    completed = 0
    failed = 0
    for entry in entries:
        status = str(entry.get("status") or "").strip().lower()
        if not status or status == "skipped":
            continue
        total += 1
        if status in {"completed", "success", "built"}:
            completed += 1
        elif status in {"failed", "error"}:
            failed += 1
    return {"total": total, "completed": completed, "failed": failed}


class NoteStyleIndexWriter:
    """Maintain the note_style index file for a run."""

    def __init__(self, *, sid: str, paths: NoteStylePaths) -> None:
        self.sid = str(sid)
        self._paths = paths
        self._index_path = paths.index_file
        self._index_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _load_document(self) -> MutableMapping[str, Any]:
        try:
            raw = self._index_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return {}
        except OSError:
            log.warning(
                "NOTE_STYLE_INDEX_READ_FAILED sid=%s path=%s",
                self.sid,
                self._index_path,
                exc_info=True,
            )
            return {}
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            log.warning(
                "NOTE_STYLE_INDEX_PARSE_FAILED sid=%s path=%s",
                self.sid,
                self._index_path,
                exc_info=True,
            )
            return {}
        if isinstance(payload, MutableMapping):
            return dict(payload)
        if isinstance(payload, Mapping):
            return dict(payload)
        return {}

    def _extract_entries(
        self, document: MutableMapping[str, Any]
    ) -> tuple[str, list[Mapping[str, Any]]]:
        for key in ("packs", "items"):
            container = document.get(key)
            if isinstance(container, Sequence):
                entries = [entry for entry in container if isinstance(entry, Mapping)]
                document[key] = list(entries)
                return key, list(entries)
        document["packs"] = []
        return "packs", []

    def _atomic_write_index(self, document: Mapping[str, Any]) -> None:
        tmp_path = self._index_path.with_suffix(
            self._index_path.suffix + f".tmp.{uuid.uuid4().hex}"
        )
        try:
            with tmp_path.open("w", encoding="utf-8", newline="") as handle:
                json.dump(document, handle, ensure_ascii=False, indent=2)
                handle.flush()
                try:
                    os.fsync(handle.fileno())
                except OSError:
                    pass
            os.replace(tmp_path, self._index_path)
        finally:
            try:
                tmp_path.unlink()
            except FileNotFoundError:
                pass
        _fsync_directory(self._index_path.parent)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def mark_completed(
        self,
        account_id: str,
        *,
        pack_path: Path | None,
        result_path: Path | None,
        completed_at: str | None = None,
        note_hash: str | None = None,
    ) -> tuple[Mapping[str, Any], dict[str, int]]:
        document = self._load_document()
        key, entries = self._extract_entries(document)

        timestamp = completed_at or _now_iso()
        normalized_account = str(account_id)

        rewritten: list[dict[str, Any]] = []
        updated_entry: dict[str, Any] | None = None
        for entry in entries:
            entry_payload = dict(entry)
            if str(entry_payload.get("account_id") or "") == normalized_account:
                entry_payload["status"] = "completed"
                entry_payload["completed_at"] = timestamp
                if result_path is not None:
                    entry_payload["result"] = _relativize(result_path, self._paths.base)
                if pack_path is not None:
                    entry_payload.setdefault(
                        "pack", _relativize(pack_path, self._paths.base)
                    )
                if note_hash:
                    entry_payload.setdefault("note_hash", note_hash)
                entry_payload.pop("error", None)
                updated_entry = entry_payload
            rewritten.append(entry_payload)

        if updated_entry is None:
            entry_payload = {
                "account_id": normalized_account,
                "status": "completed",
                "completed_at": timestamp,
            }
            if pack_path is not None:
                entry_payload["pack"] = _relativize(pack_path, self._paths.base)
            if result_path is not None:
                entry_payload["result"] = _relativize(result_path, self._paths.base)
            if note_hash:
                entry_payload["note_hash"] = note_hash
            rewritten.append(entry_payload)
            updated_entry = entry_payload

        rewritten.sort(key=lambda item: str(item.get("account_id") or ""))
        document[key] = rewritten
        document["totals"] = _compute_totals(rewritten)

        self._atomic_write_index(document)
        totals = document["totals"]
        index_relative = _relative_to_base(self._index_path, self._paths.base)
        status_text = str(updated_entry.get("status") or "") if updated_entry else ""
        pack_value = str(updated_entry.get("pack") or "") if updated_entry else ""
        result_value = str(updated_entry.get("result") or "") if updated_entry else ""
        note_hash_value = str(updated_entry.get("note_hash") or "") if updated_entry else ""
        log.info(
            "STYLE_INDEX_UPDATED sid=%s account_id=%s action=completed status=%s packs_total=%s packs_completed=%s packs_failed=%s index=%s pack=%s result=%s note_hash=%s",
            self.sid,
            normalized_account,
            status_text,
            totals.get("total", 0),
            totals.get("completed", 0),
            totals.get("failed", 0),
            index_relative,
            pack_value,
            result_value,
            note_hash_value,
        )
        return updated_entry, totals


def store_note_style_result(
    sid: str,
    account_id: str,
    payload: Mapping[str, Any],
    *,
    runs_root: Path | str | None = None,
    completed_at: str | None = None,
) -> Path:
    """Persist the model ``payload`` for ``account_id`` and update the index."""

    runs_root_path = _resolve_runs_root(runs_root)
    ensure_note_style_section(sid, runs_root=runs_root_path)
    paths = ensure_note_style_paths(runs_root_path, sid, create=True)
    account_paths = ensure_note_style_account_paths(paths, account_id, create=True)

    _atomic_write_jsonl(account_paths.result_file, payload)
    result_relative = _relative_to_base(account_paths.result_file, paths.base)
    log.info(
        "STYLE_RESULTS_WRITTEN sid=%s account_id=%s result=%s prompt_salt=%s source_hash=%s",
        sid,
        account_id,
        result_relative,
        str(payload.get("prompt_salt") or ""),
        str(payload.get("source_hash") or ""),
    )

    writer = NoteStyleIndexWriter(sid=sid, paths=paths)
    updated_entry, totals = writer.mark_completed(
        account_id,
        pack_path=account_paths.pack_file,
        result_path=account_paths.result_file,
        completed_at=completed_at,
        note_hash=str(payload.get("note_hash") or "") or None,
    )

    try:
        refresh_note_style_stage_from_index(sid, runs_root=runs_root_path)
    except Exception:  # pragma: no cover - defensive logging
        log.warning(
            "NOTE_STYLE_STAGE_REFRESH_FAILED sid=%s", sid, exc_info=True
        )
    else:
        status_text = str(updated_entry.get("status") or "") if isinstance(updated_entry, Mapping) else ""
        log.info(
            "STYLE_STAGE_REFRESH sid=%s account_id=%s stage_status=%s packs_total=%s results_completed=%s results_failed=%s",
            sid,
            account_id,
            status_text,
            totals.get("total", 0),
            totals.get("completed", 0),
            totals.get("failed", 0),
        )

    try:
        runflow_barriers_refresh(sid)
    except Exception:  # pragma: no cover - defensive logging
        log.warning(
            "NOTE_STYLE_BARRIERS_REFRESH_FAILED sid=%s", sid, exc_info=True
        )

    try:
        barrier_state = reconcile_umbrella_barriers(sid, runs_root=runs_root_path)
    except Exception:  # pragma: no cover - defensive logging
        log.warning(
            "NOTE_STYLE_BARRIERS_RECONCILE_FAILED sid=%s", sid, exc_info=True
        )
    else:
        log.info(
            "[Runflow] Umbrella barriers: sid=%s stage=note_style state=%s",
            sid,
            _structured_repr(barrier_state),
        )

    return account_paths.result_file


__all__ = ["NoteStyleIndexWriter", "store_note_style_result"]
