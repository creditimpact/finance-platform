from __future__ import annotations

from typing import Any, Dict, List, Optional
from functools import wraps

from pydantic import ValidationError

from backend.config import CASESTORE_REDACT_BEFORE_STORE
from backend.core.config.flags import FLAGS

from .errors import CaseStoreError, NOT_FOUND, VALIDATION_FAILED
from .models import Artifact, AccountCase, AccountFields, Bureau, SessionCase
from .redaction import redact_account_fields
from .merge import safe_deep_merge
from .storage import load_session_case as _load, save_session_case as _save
from .telemetry import emit, timed

__all__ = [
    "create_session_case",
    "load_session_case",
    "save_session_case",
    "upsert_account_fields",
    "get_account_case",
    "get_account_fields",
    "append_artifact",
    "set_tags",
    "list_accounts",
]


def _emit_on_error(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        session_id = kwargs.get("session_id")
        if session_id is None and args:
            first = args[0]
            session_id = getattr(first, "session_id", first)
        try:
            return fn(*args, **kwargs)
        except CaseStoreError as err:
            emit("case_store_error", session_id=session_id, code=err.code, where="api")
            raise

    return wrapper


def _count_masked_fields(original: Dict[str, Any], redacted: Dict[str, Any]) -> int:
    count = 0

    def _walk(o: Any, r: Any) -> None:
        nonlocal count
        if isinstance(o, dict) and isinstance(r, dict):
            keys = set(o.keys()) | set(r.keys())
            for k in keys:
                _walk(o.get(k), r.get(k))
        elif isinstance(o, list) and isinstance(r, list):
            for a, b in zip(o, r):
                _walk(a, b)
            if len(o) != len(r):
                count += abs(len(o) - len(r))
        else:
            if o != r:
                count += 1

    _walk(original, redacted)
    return count


def _coerce_bureau(value: str | Bureau) -> Bureau:
    if isinstance(value, Bureau):
        return value
    for member in Bureau:
        if member.value.lower() == str(value).lower():
            return member
    raise CaseStoreError(code=VALIDATION_FAILED, message=f"Invalid bureau: {value}")


def create_session_case(session_id: str, meta: Dict[str, Any] | None = None) -> SessionCase:
    """Create a new in-memory SessionCase with optional metadata."""

    case = SessionCase(session_id=session_id, accounts={})
    if meta:
        fields = case.report_meta.__class__.model_fields
        allowed = {k: v for k, v in meta.items() if k in fields}
        if allowed:
            case.report_meta = case.report_meta.model_copy(update=allowed)
    return case


@_emit_on_error
def load_session_case(session_id: str) -> SessionCase:
    """Load a session from storage."""

    return _load(session_id)


@_emit_on_error
def save_session_case(case: SessionCase) -> None:
    """Persist a session to storage."""

    _save(case)


@_emit_on_error
def upsert_account_fields(
    session_id: str,
    account_id: str,
    bureau: str | Bureau,
    fields: Dict[str, Any],
) -> None:
    """Upsert account fields, optionally redacting sensitive data."""

    bureau_enum = _coerce_bureau(bureau)
    with timed(
        "case_store_upsert",
        session_id=session_id,
        account_id=account_id,
        bureau=bureau_enum.value,
    ) as t:
        case = _load(session_id)
        if CASESTORE_REDACT_BEFORE_STORE:
            original = fields
            fields = redact_account_fields(fields)
            t.base["masked_fields_count"] = _count_masked_fields(original, fields)

        account = case.accounts.get(account_id)
        if account is None:
            account = AccountCase(bureau=bureau_enum)
            case.accounts[account_id] = account
        else:
            account.bureau = bureau_enum

        current = account.fields.model_dump()
        if FLAGS.safe_merge_enabled:
            merged = safe_deep_merge(current, fields)
        else:
            current.update(fields)
            merged = current
        try:
            account.fields = AccountFields(**merged)
        except ValidationError as exc:
            raise CaseStoreError(code=VALIDATION_FAILED, message=str(exc)) from exc

        save_session_case(case)


@_emit_on_error
def get_account_case(session_id: str, account_id: str) -> AccountCase:
    """Return the full AccountCase."""

    case = _load(session_id)
    account = case.accounts.get(account_id)
    if account is None:
        raise CaseStoreError(code=NOT_FOUND, message=f"Account '{account_id}' not found")
    return account


@_emit_on_error
def get_account_fields(
    session_id: str,
    account_id: str,
    field_names: List[str],
) -> Dict[str, Any]:
    """Return a dict subset of AccountFields by name."""

    case = _load(session_id)
    account = case.accounts.get(account_id)
    if account is None:
        raise CaseStoreError(code=NOT_FOUND, message=f"Account '{account_id}' not found")

    return {name: getattr(account.fields, name, None) for name in field_names}


@_emit_on_error
def append_artifact(
    session_id: str,
    account_id: str,
    namespace: str,
    payload: Dict[str, Any],
    *,
    overwrite: bool = True,
    attach_provenance: Optional[Dict[str, Any]] = None,
) -> None:
    """Add or replace an artifact under a namespace."""

    with timed(
        "case_store_artifact_append",
        session_id=session_id,
        account_id=account_id,
        namespace=namespace,
        overwrite=overwrite,
    ):
        case = _load(session_id)
        account = case.accounts.get(account_id)
        if account is None:
            raise CaseStoreError(code=NOT_FOUND, message=f"Account '{account_id}' not found")

        existing = account.artifacts.get(namespace)
        if not overwrite and existing is not None:
            raise CaseStoreError(code=VALIDATION_FAILED, message="Artifact exists")

        try:
            artifact = Artifact(**payload)
        except ValidationError as exc:
            raise CaseStoreError(code=VALIDATION_FAILED, message=str(exc)) from exc

        if attach_provenance:
            debug = artifact.debug or {}
            debug["provenance"] = attach_provenance
            artifact.debug = debug

        account.artifacts[namespace] = artifact
        save_session_case(case)


@_emit_on_error
def set_tags(session_id: str, account_id: str, **tags) -> None:
    """Merge tags into the account."""

    case = _load(session_id)
    account = case.accounts.get(account_id)
    if account is None:
        raise CaseStoreError(code=NOT_FOUND, message=f"Account '{account_id}' not found")

    account.tags.update(tags)
    save_session_case(case)


@_emit_on_error
def list_accounts(
    session_id: str, bureau: str | Bureau | None = None
) -> List[str]:
    """List account IDs for a session, optionally filtered by bureau."""

    case = _load(session_id)
    if bureau is None:
        return list(case.accounts.keys())

    bureau_enum = _coerce_bureau(bureau)
    return [aid for aid, acc in case.accounts.items() if acc.bureau == bureau_enum]
