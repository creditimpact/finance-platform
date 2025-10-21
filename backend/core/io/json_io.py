"""Lightweight JSON IO helpers with atomic writes."""

from __future__ import annotations

import copy
import json
import os
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping


_LOCK_POLL_INTERVAL = 0.05
_WRITE_RETRY_DELAY = 0.1
_WRITE_ATTEMPTS = 2


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


@contextmanager
def _json_file_lock(path: Path) -> Iterator[None]:
    """Serialize writers for ``path`` using a simple lock file."""

    lock_path = path.with_suffix(path.suffix + ".lock")
    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError:
            time.sleep(_LOCK_POLL_INTERVAL)
    try:
        os.close(fd)
        yield
    finally:
        try:
            os.unlink(lock_path)
        except FileNotFoundError:
            pass


def _atomic_write_json(path: Path, payload: Any) -> None:
    """Atomically write ``payload`` as JSON to ``path``."""

    path.parent.mkdir(parents=True, exist_ok=True)

    attempts = 0
    last_error: OSError | None = None
    while attempts < _WRITE_ATTEMPTS:
        attempts += 1
        tmp_path = path.with_suffix(path.suffix + f".tmp.{uuid.uuid4().hex}")
        try:
            with _json_file_lock(path):
                payload_to_write = _merge_existing_umbrella_barriers(path, payload)
                tmp_path.write_text(
                    json.dumps(payload_to_write, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                tmp_path.replace(path)
        except OSError as exc:
            last_error = exc
            try:
                tmp_path.unlink()
            except FileNotFoundError:
                pass
            if attempts >= _WRITE_ATTEMPTS:
                break
            time.sleep(_WRITE_RETRY_DELAY)
        else:
            return

    if last_error is not None:
        raise last_error


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

