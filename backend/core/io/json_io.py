"""Lightweight JSON IO helpers with atomic writes."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Callable, Mapping


def _merge_existing_umbrella_barriers(path: Path, payload: Any) -> Any:
    """Merge persisted umbrella readiness data into ``payload`` when writing."""

    if path.name != "runflow.json":
        return payload

    if not isinstance(payload, Mapping):
        return payload

    try:
        existing_raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return payload
    except OSError:
        return payload

    try:
        existing_payload = json.loads(existing_raw)
    except json.JSONDecodeError:
        return payload

    if not isinstance(existing_payload, Mapping):
        return payload

    existing_barriers = existing_payload.get("umbrella_barriers")
    if not isinstance(existing_barriers, Mapping):
        return payload

    merged_barriers = dict(existing_barriers)
    new_barriers = payload.get("umbrella_barriers")
    if isinstance(new_barriers, Mapping):
        merged_barriers.update(new_barriers)

    merged_payload = dict(payload)
    merged_payload["umbrella_barriers"] = merged_barriers
    return merged_payload


def _atomic_write_json(path: Path, payload: Any) -> None:
    """Atomically write ``payload`` as JSON to ``path``."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload_to_write = _merge_existing_umbrella_barriers(path, payload)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload_to_write, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tmp.replace(path)


def update_json_in_place(
    path: Path | str, update_fn: Callable[[Any], Any | None]
) -> Any:
    """Update ``path`` JSON atomically using ``update_fn``.

    The existing JSON payload (or ``{}`` when the file is missing) is deep-copied
    before ``update_fn`` is invoked so callers can mutate the provided object.
    When ``update_fn`` returns ``None`` the mutated value is written back. If the
    payload is unchanged the file is left untouched.
    """

    json_path = Path(path)

    try:
        raw = json_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        current_payload: Any = {}
    else:
        try:
            current_payload = json.loads(raw)
        except json.JSONDecodeError as exc:  # pragma: no cover - defensive
            raise ValueError(f"Invalid JSON content in {json_path}") from exc

    original_snapshot = copy.deepcopy(current_payload)
    working_copy = copy.deepcopy(current_payload)

    result = update_fn(working_copy)
    new_payload = working_copy if result is None else result

    if new_payload == original_snapshot:
        return new_payload

    _atomic_write_json(json_path, new_payload)
    return new_payload


__all__ = ["_atomic_write_json", "update_json_in_place"]

