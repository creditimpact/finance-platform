"""Helpers for persisting note_style model outputs.

This module will be implemented in a follow-up task. The current stub keeps the
public API importable so callers can depend on it without performing any IO
work yet.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from backend.core.ai.paths import NoteStyleAccountPaths


def ingest_note_style_result(
    *,
    sid: str,
    account_id: str,
    runs_root: Path,
    account_paths: NoteStyleAccountPaths,
    pack_payload: Mapping[str, Any],
    response_payload: Any,
) -> Path | None:
    """Persist the model ``response_payload`` for ``account_id``.

    The concrete implementation is provided in a subsequent task. The stub
    keeps the call site functional and raises :class:`NotImplementedError` so
    tests can monkeypatch it as needed.
    """

    raise NotImplementedError("ingest_note_style_result is not implemented yet")


__all__ = ["ingest_note_style_result"]
