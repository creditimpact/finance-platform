"""Helpers for reconciling runflow stage counters with filesystem artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import json

from backend.frontend.packs.config import load_frontend_stage_config


def _coerce_int(value: Any) -> Optional[int]:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _load_document(path: Path) -> Optional[Mapping[str, Any]]:
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError:
        return None

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return None

    if isinstance(payload, Mapping):
        return payload

    if isinstance(payload, Sequence):
        # Legacy layouts occasionally serialised the payload as a list.
        return {"items": list(payload)}

    return None


def validation_findings_count(base_dir: Path) -> Optional[int]:
    """Return the number of validation findings written to disk for ``sid``."""

    index_path = base_dir / "ai_packs" / "validation" / "index.json"
    document = _load_document(index_path)
    if document is None:
        return None

    total = 0
    found = False

    for key in ("packs", "items"):
        entries = document.get(key)
        if not isinstance(entries, Sequence):
            continue
        for entry in entries:
            if not isinstance(entry, Mapping):
                continue
            lines = _coerce_int(entry.get("lines"))
            if lines is None:
                lines = _coerce_int(entry.get("line_count"))
            if lines is not None:
                total += lines
            found = True

    if found:
        return total

    totals = document.get("totals")
    if isinstance(totals, Mapping):
        for candidate in ("findings", "weak_count", "fields_built", "count"):
            value = _coerce_int(totals.get(candidate))
            if value is not None:
                return value

    fallback = _coerce_int(document.get("findings_count"))
    if fallback is not None:
        return fallback

    return None


def frontend_packs_count(base_dir: Path) -> Optional[int]:
    """Return the number of frontend review packs written for ``sid``."""

    config = load_frontend_stage_config(base_dir)
    packs_dir = config.packs_dir

    if not packs_dir.exists() or not packs_dir.is_dir():
        return None

    try:
        return sum(
            1
            for entry in packs_dir.iterdir()
            if entry.is_file() and entry.suffix == ".json"
        )
    except OSError:
        return None


def merge_scored_pairs_count(base_dir: Path) -> Optional[int]:
    """Return the number of merge pairs scored for ``sid``."""

    index_path = base_dir / "ai_packs" / "merge" / "pairs_index.json"
    document = _load_document(index_path)
    if document is None:
        return None

    totals = document.get("totals")
    if isinstance(totals, Mapping):
        value = _coerce_int(totals.get("scored_pairs"))
        if value is not None:
            return value

    fallback = _coerce_int(document.get("scored_pairs"))
    if fallback is not None:
        return fallback

    return None


def stage_counts(stage: str, base_dir: Path) -> dict[str, int]:
    """Return authoritative counter mappings for ``stage`` rooted at ``base_dir``."""

    stage_key = str(stage)
    if stage_key == "validation":
        value = validation_findings_count(base_dir)
        return {"findings_count": value} if value is not None else {}
    if stage_key == "frontend":
        value = frontend_packs_count(base_dir)
        return {"packs_count": value} if value is not None else {}
    if stage_key == "merge":
        value = merge_scored_pairs_count(base_dir)
        return {"scored_pairs": value} if value is not None else {}
    return {}


__all__ = [
    "frontend_packs_count",
    "merge_scored_pairs_count",
    "stage_counts",
    "validation_findings_count",
]
