from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import ValidationError

from backend.config import CASESTORE_REDACT_BEFORE_STORE

from .errors import CaseStoreError, NOT_FOUND, VALIDATION_FAILED
from .models import Artifact, AccountCase, AccountFields, Bureau, SessionCase
from .redaction import redact_account_fields
from .storage import load_session_case as _load, save_session_case as _save

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


def load_session_case(session_id: str) -> SessionCase:
    """Load a session from storage."""

    return _load(session_id)


def save_session_case(case: SessionCase) -> None:
    """Persist a session to storage."""

    _save(case)


def upsert_account_fields(
    session_id: str,
    account_id: str,
    bureau: str | Bureau,
    fields: Dict[str, Any],
) -> None:
    """Upsert account fields, optionally redacting sensitive data."""

    case = _load(session_id)
    if CASESTORE_REDACT_BEFORE_STORE:
        fields = redact_account_fields(fields)

    bureau_enum = _coerce_bureau(bureau)

    account = case.accounts.get(account_id)
    if account is None:
        account = AccountCase(bureau=bureau_enum)
        case.accounts[account_id] = account
    else:
        account.bureau = bureau_enum

    current = account.fields.model_dump()
    current.update(fields)
    try:
        account.fields = AccountFields(**current)
    except ValidationError as exc:
        raise CaseStoreError(code=VALIDATION_FAILED, message=str(exc)) from exc

    save_session_case(case)


def get_account_case(session_id: str, account_id: str) -> AccountCase:
    """Return the full AccountCase."""

    case = _load(session_id)
    account = case.accounts.get(account_id)
    if account is None:
        raise CaseStoreError(code=NOT_FOUND, message=f"Account '{account_id}' not found")
    return account


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


def set_tags(session_id: str, account_id: str, **tags) -> None:
    """Merge tags into the account."""

    case = _load(session_id)
    account = case.accounts.get(account_id)
    if account is None:
        raise CaseStoreError(code=NOT_FOUND, message=f"Account '{account_id}' not found")

    account.tags.update(tags)
    save_session_case(case)


def list_accounts(
    session_id: str, bureau: str | Bureau | None = None
) -> List[str]:
    """List account IDs for a session, optionally filtered by bureau."""

    case = _load(session_id)
    if bureau is None:
        return list(case.accounts.keys())

    bureau_enum = _coerce_bureau(bureau)
    return [aid for aid, acc in case.accounts.items() if acc.bureau == bureau_enum]
