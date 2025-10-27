"""Helpers for normalizing note_style stage filesystem paths."""

from __future__ import annotations

import re
from pathlib import Path


_WINDOWS_DRIVE_PATTERN = re.compile(r"^[A-Za-z]:")


def _normalize_path_for_worker(run_root: Path, raw: str) -> Path:
    """Return a resolved path for a worker given a manifest or ENV value."""

    run_root_path = Path(run_root).resolve()
    sanitized = str(raw or "").strip()
    if not sanitized:
        raise ValueError("path value must be a non-empty string")

    sanitized = sanitized.replace("\\", "/")
    drive_match = _WINDOWS_DRIVE_PATTERN.match(sanitized)
    drive_stripped = False
    if drive_match:
        sanitized = sanitized[drive_match.end():]
        drive_stripped = True

    candidate = Path(sanitized)

    if candidate.is_absolute():
        if drive_stripped:
            parts = [part for part in candidate.parts if part not in {"", ".", "/"}]
            lowered = [part.lower() for part in parts]
            run_root_name = run_root_path.name.lower()
            if run_root_name and run_root_name in lowered:
                idx = lowered.index(run_root_name)
                parts = parts[idx + 1 :]
            candidate = run_root_path.joinpath(*parts)
        try:
            return candidate.resolve()
        except OSError:
            return candidate

    candidate = run_root_path / candidate

    try:
        return candidate.resolve()
    except OSError:
        return candidate
