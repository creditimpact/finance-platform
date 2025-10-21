"""Helpers for reconciling runflow stage counters with filesystem artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

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


def _has_review_attachments(payload: Mapping[str, Any]) -> bool:
    attachments = payload.get("attachments")
    if isinstance(attachments, Mapping):
        for value in attachments.values():
            if isinstance(value, str) and value.strip():
                return True
            if isinstance(value, Iterable) and not isinstance(
                value, (str, bytes, bytearray)
            ):
                for entry in value:
                    if isinstance(entry, str) and entry.strip():
                        return True

    legacy = payload.get("evidence")
    if isinstance(legacy, Iterable) and not isinstance(legacy, (str, bytes, bytearray)):
        for item in legacy:
            if not isinstance(item, Mapping):
                continue
            docs = item.get("docs")
            if isinstance(docs, Iterable) and not isinstance(
                docs, (str, bytes, bytearray)
            ):
                for doc in docs:
                    if isinstance(doc, Mapping):
                        doc_ids = doc.get("doc_ids")
                        if isinstance(doc_ids, Iterable) and not isinstance(
                            doc_ids, (str, bytes, bytearray)
                        ):
                            for doc_id in doc_ids:
                                if isinstance(doc_id, str) and doc_id.strip():
                                    return True
    return False


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


def frontend_answers_counters(
    base_dir: Path,
    *,
    attachments_required: bool,
) -> dict[str, int]:
    """Return frontend response answer counters rooted at ``base_dir``."""

    required = frontend_packs_count(base_dir) or 0

    config = load_frontend_stage_config(base_dir)
    responses_dir = config.responses_dir

    try:
        entries = sorted(
            path
            for path in responses_dir.iterdir()
            if path.is_file() and path.name.endswith(".result.json")
        )
    except OSError:
        entries = []

    answered_ids: set[str] = set()

    for entry in entries:
        payload = _load_document(entry)
        if payload is None:
            continue

        answers = payload.get("answers")
        if not isinstance(answers, Mapping):
            continue

        explanation = answers.get("explanation")
        if not isinstance(explanation, str) or not explanation.strip():
            continue

        if attachments_required and not _has_review_attachments(answers):
            continue

        received_at = payload.get("received_at")
        if not isinstance(received_at, str) or not received_at.strip():
            continue

        account_id = payload.get("account_id")
        if isinstance(account_id, str) and account_id.strip():
            answered_ids.add(account_id.strip())
        else:
            answered_ids.add(entry.stem)

    return {
        "answers_required": required,
        "answers_received": len(answered_ids),
    }


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
    "frontend_answers_counters",
    "frontend_packs_count",
    "merge_scored_pairs_count",
    "stage_counts",
    "validation_findings_count",
]
