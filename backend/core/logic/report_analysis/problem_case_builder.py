"""Materialise per-account problem case files.

This module exposes :func:`build_problem_cases` which, given a list of
candidate accounts, writes one JSON file per account under
``runs/<sid>/cases/accounts`` and creates a summary ``index.json`` for the
session.  The builder is intentionally dumb – it does not attempt to
determine which accounts are problematic; that decision must be supplied
via ``candidates``.

The function returns a small summary dictionary describing the work
performed so callers can easily log or inspect the results.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

from backend.pipeline.runs import RunManifest, write_breadcrumb
from backend.settings import PROJECT_ROOT

from .keys import compute_logical_account_key

logger = logging.getLogger(__name__)


def _make_account_id(account: Mapping[str, Any], idx: int) -> str:
    """Return a filesystem‑friendly identifier for ``account``.

    The builder needs a way to match candidate ``account_id`` values with
    the accounts loaded from ``accounts_from_full.json``.  When possible we
    compute a logical key using identifying fields; otherwise a deterministic
    surrogate based on the heading or list position is used.
    """

    fields = account.get("fields") or {}

    issuer = (
        fields.get("issuer")
        or fields.get("creditor")
        or fields.get("name")
        or account.get("issuer")
        or account.get("creditor")
        or account.get("name")
    )
    last4 = (
        fields.get("account_last4")
        or fields.get("last4")
        or account.get("account_last4")
        or account.get("last4")
    )
    account_type = (
        fields.get("account_type")
        or fields.get("type")
        or account.get("account_type")
        or account.get("type")
    )
    opened_date = (
        fields.get("opened_date")
        or fields.get("date_opened")
        or account.get("opened_date")
        or account.get("date_opened")
    )

    logical_key = compute_logical_account_key(issuer, last4, account_type, opened_date)
    acc_id = logical_key or f"idx-{idx:03d}"
    return re.sub(r"[^a-z0-9_-]", "_", str(acc_id).lower())


def _load_accounts(path: Path) -> List[Mapping[str, Any]]:
    """Return account dictionaries from ``accounts_from_full.json``.

    The file can either contain a JSON object with an ``accounts`` key or a
    bare list.  Errors are swallowed and result in an empty list.
    """

    if not path.exists():
        return []

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []

    if isinstance(data, Mapping):
        accounts = data.get("accounts") or []
    elif isinstance(data, list):
        accounts = data
    else:
        accounts = []

    out: List[Mapping[str, Any]] = []
    for acc in accounts:
        if isinstance(acc, Mapping):
            out.append(acc)
    return out


def _build_account_lookup(
    accounts: Iterable[Mapping[str, Any]]
) -> Dict[str, Mapping[str, Any]]:
    """Build a mapping of possible identifiers to account records."""

    by_key: Dict[str, Mapping[str, Any]] = {}
    for idx, acc in enumerate(accounts, start=1):
        account_index = acc.get("account_index")
        if isinstance(account_index, int):
            by_key[str(account_index)] = acc

        acc_id = _make_account_id(acc, idx)
        by_key[acc_id] = acc

    return by_key


def build_problem_cases(
    sid: str, candidates: List[Dict[str, Any]] | None = None, root: Path | None = None
) -> Dict[str, Any]:
    """Materialise problem case files for ``sid``.

    Parameters
    ----------
    sid:
        Session identifier used to locate trace artefacts.
    candidates:
        Optional list describing problematic accounts.  Each item must include
        ``account_id`` and may optionally include ``account_index`` and
        ``confidence``.  If omitted, an empty list is assumed.
    root:
        Repository root; defaults to :data:`backend.settings.PROJECT_ROOT`.
    """

    root_path = Path(root or PROJECT_ROOT)
    # Prefer manifest accounts_json if available; fallback to legacy path
    try:
        m_probe = RunManifest.for_sid(sid)
        acc_path = Path(m_probe.get("traces.accounts_table", "accounts_json"))
    except Exception:
        acc_path = (
            root_path
            / "traces"
            / "blocks"
            / sid
            / "accounts_table"
            / "accounts_from_full.json"
        )

    accounts = _load_accounts(acc_path)
    total = len(accounts)
    lookup = _build_account_lookup(accounts)

    # Register the cases directory under the run root and ensure it exists
    m = RunManifest.for_sid(sid)
    cases_dir = m.ensure_run_subdir("cases_dir", "cases")
    accounts_dir = cases_dir / "accounts"
    accounts_dir.mkdir(parents=True, exist_ok=True)
    m.set_artifact("cases", "case_dir", cases_dir)
    write_breadcrumb(m.path, cases_dir / ".manifest")

    logger.info("PROBLEM_CASES start sid=%s total=%s out=%s", sid, total, cases_dir)

    for cand in candidates or []:
        if not isinstance(cand, Mapping):
            continue

        account_id = cand.get("account_id")
        if account_id is None:
            continue

        # Locate full account record either by provided index or account_id
        full_acc: Mapping[str, Any] | None = None
        if isinstance(cand.get("account_index"), int):
            full_acc = lookup.get(str(cand["account_index"]))
        if full_acc is None:
            full_acc = lookup.get(str(account_id), {})

        payload = {
            "sid": sid,
            "account_id": account_id,
            "problem_tags": cand.get("problem_tags") or [],
            "problem_reasons": cand.get("problem_reasons") or [],
            "confidence": cand.get("confidence"),
            "account": full_acc or {},
        }

        (accounts_dir / f"{account_id}.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    cand_list = list(candidates or [])
    index_data = {
        "sid": sid,
        "total": total,
        "problematic": len(cand_list),
        "problematic_accounts": [c.get("account_id") for c in cand_list],
    }
    (cases_dir / "index.json").write_text(
        json.dumps(index_data, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    logger.info(
        "PROBLEM_CASES done sid=%s total=%s problematic=%s out=%s",
        sid,
        total,
        len(cand_list),
        cases_dir,
    )

    return {
        "sid": sid,
        "total": total,
        "problematic": len(cand_list),
        "out": str(cases_dir),
    }


__all__ = ["build_problem_cases"]
