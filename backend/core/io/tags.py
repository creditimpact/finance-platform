"""Helper functions for managing account tags."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable, List, Sequence


def read_tags(path: os.PathLike | str) -> List[dict]:
    """Read tags from ``path``.

    Returns an empty list when the file does not exist.
    """

    tag_path = Path(path)
    if not tag_path.exists():
        return []

    with tag_path.open("r", encoding="utf-8") as file:
        try:
            data = json.load(file)
        except json.JSONDecodeError as exc:  # pragma: no cover - defensive
            raise ValueError(f"Invalid JSON in tags file: {tag_path}") from exc

    if not isinstance(data, list):
        raise ValueError(f"Expected list in tags file: {tag_path}")

    return data


def upsert_tag(path: os.PathLike | str, tag: dict, key_fields: Sequence[str]) -> None:
    """Insert or update ``tag`` at ``path`` keyed by ``key_fields``.

    When an existing tag shares the same key values, it is replaced with the new
    ``tag`` payload. Otherwise the tag is appended. The write is performed
    atomically.
    """

    tags = read_tags(path)

    def matches(candidate: dict) -> bool:
        return all(candidate.get(field) == tag.get(field) for field in key_fields)

    updated = False
    for index, existing in enumerate(tags):
        if matches(existing):
            tags[index] = tag
            updated = True
            break

    if not updated:
        tags.append(tag)

    write_tags(path, tags)


def write_tags(path: os.PathLike | str, tags: Iterable[dict]) -> None:
    """Write ``tags`` to ``path`` atomically."""

    tag_path = Path(path)
    tag_path.parent.mkdir(parents=True, exist_ok=True)

    temp_path = tag_path.with_suffix(tag_path.suffix + ".tmp")

    # Ensure deterministic output for re-runs.
    data = list(tags)

    with temp_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2, sort_keys=True)
        file.flush()
        os.fsync(file.fileno())

    os.replace(temp_path, tag_path)
