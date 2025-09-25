"""Lightweight JSON IO helpers with atomic writes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _atomic_write_json(path: Path, payload: Any) -> None:
    """Atomically write ``payload`` as JSON to ``path``."""

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tmp.replace(path)


__all__ = ["_atomic_write_json"]

