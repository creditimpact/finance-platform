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
                existing[pack_path] = payload

        ordered = sorted(
            existing.values(),
            key=lambda item: (
                _safe_int(item.get("account_id")),
                str(item.get("pack_path") or ""),
            ),
        )

        document = {
            "schema_version": _SCHEMA_VERSION,
            "sid": self.sid,
            "packs": ordered,
        }

        self._write_index(document)

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


def _safe_int(value: object) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0

