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


def _resolve_inputs_from_manifest(sid: str) -> tuple[Path, Path, RunManifest]:
    m = RunManifest.for_sid(sid)
    try:
        accounts_path = Path(m.get("traces.accounts_table", "accounts_json")).resolve()
        general_path = Path(m.get("traces.accounts_table", "general_json")).resolve()
    except KeyError as e:
        raise RuntimeError(f"Run manifest missing traces.accounts_table key for sid={sid}: {e}")
    return accounts_path, general_path, m


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

    acc_path, gen_path, _ = _resolve_inputs_from_manifest(sid)
    # Log canonical analyzer inputs
    logger.info(
        "ANALYZER_INPUT sid=%s accounts_json=%s general_json=%s",
        sid,
        acc_path,
        gen_path,
    )
    if not acc_path.exists():
        raise RuntimeError(
            f"accounts_from_full.json missing sid={sid} path={acc_path}"
        )

    accounts = _load_accounts(acc_path)
    total = len(accounts)
    lookup = _build_account_lookup(accounts)

    # Optional enrichment: load general_info_from_full.json once to attach
    # high-level metadata (e.g., client and report dates) to each summary.
    general_info: Dict[str, Any] | None = None
    try:
        if gen_path and gen_path.exists():
            general_info = json.loads(gen_path.read_text(encoding="utf-8"))
    except Exception:
        general_info = None  # best-effort only

    # Register the cases directory under the run root and ensure it exists
    m = RunManifest.for_sid(sid)
    cases_dir = m.ensure_run_subdir("cases_dir", "cases")
    accounts_dir = (cases_dir / "accounts").resolve()
    accounts_dir.mkdir(parents=True, exist_ok=True)
    # Register canonical case dirs and indexes for discoverability
    m.set_base_dir("cases_accounts_dir", accounts_dir)
    write_breadcrumb(m.path, cases_dir / ".manifest")
    logger.info("CASES_OUT sid=%s accounts_dir=%s", sid, accounts_dir)

    logger.info("PROBLEM_CASES start sid=%s total=%s out=%s", sid, total, cases_dir)

    written_ids: List[str] = []
    merge_groups: Dict[str, str] = {}
    for cand in candidates or []:
        if not isinstance(cand, Mapping):
            continue

        account_id = cand.get("account_id")
        if account_id is None:
            continue

        account_id_str = str(account_id)

        # Locate full account record either by provided index or account_id
        full_acc: Mapping[str, Any] | None = None
        if isinstance(cand.get("account_index"), int):
            full_acc = lookup.get(str(cand["account_index"]))
        if full_acc is None:
            full_acc = lookup.get(str(account_id), {})

        # Per-account directory and files
        acc_dir = (accounts_dir / account_id_str).resolve()
        acc_dir.mkdir(parents=True, exist_ok=True)

        # account.json: full record for that account
        account_obj: Dict[str, Any] = dict(full_acc or {})
        (acc_dir / "account.json").write_text(
            json.dumps(account_obj, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        # summary.json: reasons/tags/confidence
        summary_obj: Dict[str, Any] = {
            "sid": sid,
            "account_id": account_id,
            "problem_tags": cand.get("problem_tags") or [],
            "problem_reasons": cand.get("problem_reasons") or [],
        }
        if "confidence" in cand and cand.get("confidence") is not None:
            summary_obj["confidence"] = cand.get("confidence")
        merge_tag = cand.get("merge_tag")
        if isinstance(merge_tag, Mapping):
            try:
                merge_tag_obj = json.loads(
                    json.dumps(merge_tag, ensure_ascii=False)
                )
            except TypeError:
                merge_tag_obj = dict(merge_tag)
            summary_obj["merge_tag"] = merge_tag_obj
            group_id = merge_tag_obj.get("group_id")
            if isinstance(group_id, str):
                merge_groups[account_id_str] = group_id
        # Attach selected general metadata if available
        if isinstance(general_info, dict):
            extra: Dict[str, Any] = {}
            for k in (
                "client",
                "client_name",
                "report_date",
                "report_start",
                "report_end",
                "generated_at",
                "provider",
                "source",
            ):
                v = general_info.get(k)
                if v is not None:
                    extra[k] = v
            if extra:
                summary_obj["general"] = extra
        (acc_dir / "summary.json").write_text(
            json.dumps(summary_obj, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        # Optional discoverability per account in manifest
        try:
            m.set_artifact(f"cases.accounts.{account_id}", "dir", acc_dir)
        except Exception:
            pass

        written_ids.append(account_id_str)

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
    # Manifest artifacts
    m.set_artifact("cases", "accounts_index", accounts_dir / "index.json")
    m.set_artifact("cases", "problematic_ids", cases_dir / "index.json")

    # Build accounts/accounts index.json
    acc_index = {
        "sid": sid,
        "count": len(written_ids),
        "ids": written_ids,
        "items": [],
    }
    for aid in written_ids:
        item = {"id": aid, "dir": str((accounts_dir / aid).resolve())}
        group_id = merge_groups.get(aid)
        if group_id is not None:
            item["merge_group_id"] = group_id
        acc_index["items"].append(item)
    accounts_index_path = accounts_dir / "index.json"
    accounts_index_path.write_text(
        json.dumps(acc_index, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info(
        "CASES_INDEX sid=%s file=%s count=%d", sid, accounts_index_path, len(written_ids)
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
