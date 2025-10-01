"""Helpers to maintain the validation AI pack index."""

from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence

log = logging.getLogger(__name__)

_SCHEMA_VERSION = 1


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


@dataclass(frozen=True)
class ValidationIndexEntry:
    """Single entry describing a validation pack/result pair."""

    account_id: int
    pack_path: Path
    result_path: Path
    weak_fields: Sequence[str]
    line_count: int
    status: str
    built_at: str | None = None
    request_lines: int | None = None
    model: str | None = None
    sent_at: str | None = None
    completed_at: str | None = None
    error: str | None = None

    def to_json_payload(self) -> dict[str, object]:
        weak_fields = [str(field) for field in self.weak_fields if str(field).strip()]
        payload: dict[str, object] = {
            "account_id": int(self.account_id),
            "pack_path": str(self.pack_path.resolve()),
            "result_path": str(self.result_path.resolve()),
            "weak_fields": weak_fields,
            "lines": int(self.line_count),
            "built_at": str(self.built_at or _utc_now()),
            "status": str(self.status or "built"),
        }
        if self.request_lines is not None:
            try:
                payload["request_lines"] = int(self.request_lines)
            except (TypeError, ValueError):
                payload.pop("request_lines", None)
        if self.model is not None:
            payload["model"] = str(self.model)
        if self.sent_at:
            payload["sent_at"] = str(self.sent_at)
        if self.completed_at:
            payload["completed_at"] = str(self.completed_at)
        if self.error:
            payload["error"] = str(self.error)
        return payload


class ValidationPackIndexWriter:
    """Maintain the consolidated validation pack index file."""

    def __init__(self, *, sid: str, index_path: Path) -> None:
        self.sid = str(sid)
        self._index_path = Path(index_path)
        self._index_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def upsert(self, entry: ValidationIndexEntry) -> None:
        self.bulk_upsert([entry])

    def bulk_upsert(self, entries: Iterable[ValidationIndexEntry]) -> None:
        payloads = [entry.to_json_payload() for entry in entries]
        if not payloads:
            return

        document = self._load_index()
        existing: dict[str, dict[str, object]] = {}

        for pack in document.get("packs", []):
            if not isinstance(pack, Mapping):
                continue
            pack_path = str(pack.get("pack_path") or "").strip()
            if not pack_path:
                continue
            existing[pack_path] = dict(pack)

        for payload in payloads:
            pack_path = payload.get("pack_path")
            if isinstance(pack_path, str) and pack_path:
                existing[pack_path] = dict(payload)

        ordered = self._sort_entries(existing.values())

        document = {
            "schema_version": _SCHEMA_VERSION,
            "sid": self.sid,
            "packs": ordered,
        }

        self._write_index(document)

    def mark_sent(
        self,
        pack_path: Path | str,
        *,
        request_lines: int | None = None,
        model: str | None = None,
    ) -> dict[str, object] | None:
        """Update the index entry for ``pack_path`` to ``sent`` status."""

        set_values: dict[str, object] = {
            "status": "sent",
            "sent_at": _utc_now(),
        }
        if request_lines is not None:
            try:
                set_values["request_lines"] = int(request_lines)
            except (TypeError, ValueError):
                pass
        if model is not None:
            set_values["model"] = str(model)

        return self._update_entry_fields(
            Path(pack_path),
            set_values=set_values,
            remove_keys=("completed_at", "error"),
        )

    def record_result(
        self,
        pack_path: Path | str,
        *,
        status: str,
        error: str | None = None,
        request_lines: int | None = None,
        model: str | None = None,
        completed_at: str | None = None,
    ) -> dict[str, object] | None:
        """Persist the final status for ``pack_path`` in the index."""

        normalized_status = str(status).strip().lower()
        if normalized_status not in {"done", "error"}:
            raise ValueError("status must be 'done' or 'error'")

        set_values: dict[str, object] = {
            "status": normalized_status,
            "completed_at": completed_at or _utc_now(),
        }

        if request_lines is not None:
            try:
                set_values["request_lines"] = int(request_lines)
            except (TypeError, ValueError):
                pass

        if model is not None:
            set_values["model"] = str(model)

        remove_keys: tuple[str, ...]
        if normalized_status == "error":
            if error:
                set_values["error"] = str(error)
            else:
                set_values.setdefault("error", "unknown")
            remove_keys = ()
        else:
            remove_keys = ("error",)

        return self._update_entry_fields(
            Path(pack_path),
            set_values=set_values,
            remove_keys=remove_keys,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _load_index(self) -> dict[str, object]:
        try:
            raw_text = self._index_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return {"schema_version": _SCHEMA_VERSION, "sid": self.sid, "packs": []}
        except OSError:
            log.warning("VALIDATION_INDEX_READ_FAILED path=%s", self._index_path, exc_info=True)
            return {"schema_version": _SCHEMA_VERSION, "sid": self.sid, "packs": []}

        try:
            payload = json.loads(raw_text)
        except json.JSONDecodeError:
            log.warning(
                "VALIDATION_INDEX_INVALID_JSON path=%s", self._index_path, exc_info=True
            )
            return {"schema_version": _SCHEMA_VERSION, "sid": self.sid, "packs": []}

        if not isinstance(payload, Mapping):
            log.warning(
                "VALIDATION_INDEX_INVALID_TYPE path=%s type=%s",
                self._index_path,
                type(payload).__name__,
            )
            return {"schema_version": _SCHEMA_VERSION, "sid": self.sid, "packs": []}

        return dict(payload)

    def _write_index(self, document: Mapping[str, object]) -> None:
        try:
            serialized = json.dumps(document, ensure_ascii=False, indent=2)
        except TypeError:
            log.exception("VALIDATION_INDEX_SERIALIZE_FAILED path=%s", self._index_path)
            return

        tmp_fd, tmp_name = tempfile.mkstemp(
            prefix=f".{self._index_path.name}.", dir=str(self._index_path.parent)
        )
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as handle:
                handle.write(serialized)
                handle.write("\n")
            os.replace(tmp_name, self._index_path)
        except OSError:
            log.warning(
                "VALIDATION_INDEX_WRITE_FAILED path=%s", self._index_path, exc_info=True
            )
            try:
                os.unlink(tmp_name)
            except FileNotFoundError:
                pass
        else:
            try:
                os.unlink(tmp_name)
            except FileNotFoundError:
                pass

    def _sort_entries(
        self, entries: Iterable[Mapping[str, object]]
    ) -> list[dict[str, object]]:
        normalized: list[dict[str, object]] = []
        for entry in entries:
            if isinstance(entry, Mapping):
                normalized.append(dict(entry))

        normalized.sort(
            key=lambda item: (
                _safe_int(item.get("account_id")),
                str(item.get("pack_path") or ""),
            )
        )
        return normalized

    def _update_entry_fields(
        self,
        pack_path: Path,
        *,
        set_values: Mapping[str, object],
        remove_keys: Iterable[str] = (),
    ) -> dict[str, object] | None:
        document = self._load_index()
        packs_raw = document.get("packs")
        if not isinstance(packs_raw, Sequence):
            packs_raw = []

        target_path = str(pack_path.resolve())
        updated_entry: dict[str, object] | None = None
        next_entries: list[dict[str, object]] = []

        for entry in packs_raw:
            if not isinstance(entry, Mapping):
                continue

            entry_copy = dict(entry)
            entry_path = str(entry_copy.get("pack_path") or "").strip()
            if entry_path == target_path:
                for key in remove_keys:
                    entry_copy.pop(key, None)
                for key, value in set_values.items():
                    if value is None:
                        entry_copy.pop(key, None)
                    else:
                        entry_copy[key] = value
                updated_entry = dict(entry_copy)

            next_entries.append(entry_copy)

        if updated_entry is None:
            return None

        document["schema_version"] = int(document.get("schema_version", _SCHEMA_VERSION))
        document["sid"] = str(document.get("sid") or self.sid)
        document["packs"] = self._sort_entries(next_entries)
        self._write_index(document)
        return updated_entry


def _safe_int(value: object) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0

