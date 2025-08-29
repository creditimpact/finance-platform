from __future__ import annotations

import os
import re
from typing import Any

from backend.assets.paths import account_full_path, traces_accounts_full_dir
from backend.core.utils.atomic_io import atomic_write_json
from backend.core.utils.json_sanitize import to_json_safe
import logging

logger = logging.getLogger(__name__)


def _slug(s: str | None) -> str:
    s = (s or "").strip().lower().replace(" ", "_")
    return re.sub(r"[^a-z0-9_\-]", "", s) or "account"


def _find_nonserializable_paths(obj: Any, prefix: str = "") -> list[str]:
    paths: list[str] = []
    try:
        import json  # local import to minimize overhead
        json.dumps(obj)
        return paths  # already serializable
    except Exception:
        pass
    if isinstance(obj, dict):
        for k, v in obj.items():
            paths.extend(_find_nonserializable_paths(v, f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(obj, (list, tuple, set)):
        for i, v in enumerate(obj):
            paths.extend(_find_nonserializable_paths(v, f"{prefix}[{i}]"))
    else:
        # Mark the leaf as problematic
        paths.append(prefix or "<root>")
    return paths


def write_account_full(session_id: str, account: dict[str, Any]) -> str:
    """Write a full account JSON atomically to accounts_full directory.

    Returns the file path written.
    """

    os.makedirs(traces_accounts_full_dir(session_id), exist_ok=True)
    # Ensure we have stable id/slug
    name_for_slug = account.get("normalized_name") or account.get("name")
    # Proper fix: account_id should already be slug; fallback to slug(name)
    acc_id = str(account.get("account_id") or _slug(name_for_slug))
    # Use fingerprint for uniqueness in filename when available; fallback to slug(name)
    fingerprint = str(account.get("account_fingerprint") or "").strip()
    slug = fingerprint or _slug(name_for_slug)
    if not acc_id:
        logger.warning("write_account_full_skipped_missing_id session=%s", session_id)
        raise ValueError("missing account identifier")
    path = account_full_path(session_id, acc_id, slug)
    # Sanitize before writing to avoid non-serializable types
    safe_doc = to_json_safe(account)
    try:
        atomic_write_json(path, safe_doc, ensure_ascii=False)
    except Exception as exc:
        try:
            offenders = _find_nonserializable_paths(account)
            if offenders:
                logger.warning(
                    "writer_detect_nonserializable session=%s id=%s paths=%s",
                    session_id,
                    acc_id,
                    offenders[:10],
                )
        except Exception:
            pass
        raise
    return path
