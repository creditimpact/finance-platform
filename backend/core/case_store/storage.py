import json
import os
import random
from typing import Any

from pydantic import ValidationError

from backend.config import (
    CASESTORE_ATOMIC_WRITES,
    CASESTORE_DIR,
    CASESTORE_VALIDATE_ON_LOAD,
)

from .errors import IO_ERROR, NOT_FOUND, VALIDATION_FAILED, CaseStoreError
from .models import SessionCase
from .telemetry import emit, timed

__all__ = ["load_session_case", "save_session_case"]


def _session_path(session_id: str) -> str:
    return os.path.join(CASESTORE_DIR, f"{session_id}.json")


def load_session_case(session_id: str) -> SessionCase:
    """Load a SessionCase from disk."""
    path = _session_path(session_id)
    try:
        with timed("case_store_load", session_id=session_id) as t:
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    content = fh.read()
            except FileNotFoundError as exc:  # pragma: no cover - exercised in tests
                raise CaseStoreError(code=NOT_FOUND, message=str(exc)) from exc
            except (PermissionError, OSError, IOError) as exc:  # pragma: no cover
                raise CaseStoreError(code=IO_ERROR, message=str(exc)) from exc

            try:
                if CASESTORE_VALIDATE_ON_LOAD:
                    result = SessionCase.model_validate_json(content)
                else:
                    data: Any = json.loads(content)
                    if not isinstance(data, dict):
                        raise TypeError("SessionCase JSON must be an object")
                    result = SessionCase.model_construct(**data)
            except json.JSONDecodeError as exc:  # pragma: no cover - exercised in tests
                raise CaseStoreError(code=VALIDATION_FAILED, message=str(exc)) from exc
            except (ValidationError, TypeError) as exc:  # pragma: no cover
                raise CaseStoreError(code=VALIDATION_FAILED, message=str(exc)) from exc

            t.base["file_bytes"] = os.path.getsize(path)
            return result
    except CaseStoreError as err:
        emit("case_store_error", session_id=session_id, code=err.code, where="storage")
        raise


def save_session_case(case: SessionCase) -> None:
    """Persist a SessionCase to disk."""
    path = _session_path(case.session_id)
    data = case.model_dump(mode="json")
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))

    try:
        with timed("case_store_save", session_id=case.session_id) as t:
            try:
                os.makedirs(CASESTORE_DIR, exist_ok=True)
            except (PermissionError, OSError, IOError) as exc:  # pragma: no cover
                raise CaseStoreError(code=IO_ERROR, message=str(exc)) from exc

            if CASESTORE_ATOMIC_WRITES:
                tmp_path = f"{path}.tmp.{os.getpid()}.{random.randint(0, 1_000_000)}"
                try:
                    with open(tmp_path, "w", encoding="utf-8") as fh:
                        fh.write(payload)
                        fh.flush()
                        os.fsync(fh.fileno())
                    os.replace(tmp_path, path)
                except (PermissionError, OSError, IOError) as exc:
                    try:
                        if os.path.exists(tmp_path):
                            os.remove(tmp_path)
                    finally:  # pragma: no cover
                        raise CaseStoreError(code=IO_ERROR, message=str(exc)) from exc
            else:
                try:
                    with open(path, "w", encoding="utf-8") as fh:
                        fh.write(payload)
                except (PermissionError, OSError, IOError) as exc:  # pragma: no cover
                    raise CaseStoreError(code=IO_ERROR, message=str(exc)) from exc

            t.base["file_bytes"] = os.path.getsize(path)
    except CaseStoreError as err:
        emit("case_store_error", session_id=case.session_id, code=err.code, where="storage")
        raise
