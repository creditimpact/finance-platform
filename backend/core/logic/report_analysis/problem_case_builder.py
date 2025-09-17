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
from typing import Any, Dict, Iterable, List, Mapping, Tuple

from backend.pipeline.runs import RunManifest, write_breadcrumb

from .keys import compute_logical_account_key
from .problem_extractor import build_rule_fields_from_triad

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


def _load_summary_json(path: Path) -> Tuple[Dict[str, Any], bool]:
    """Best-effort load of ``summary.json`` contents.

    Returns a tuple containing the parsed dictionary and a flag indicating
    whether a valid mapping was loaded.  When parsing fails the dictionary is
    empty and the flag is ``False``.
    """

    if not path.exists():
        return {}, False

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}, False

    if isinstance(data, Mapping):
        return dict(data), True

    return {}, False


def _coerce_list(value: Any) -> List[Any] | None:
    if value is None:
        return None
    if isinstance(value, list):
        return list(value)
    if isinstance(value, tuple):
        return list(value)
    return [value]


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
    accounts_by_index: Dict[int, Mapping[str, Any]] = {}
    for acc in accounts:
        idx = acc.get("account_index")
        if isinstance(idx, int):
            accounts_by_index[idx] = acc

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
        account_index = cand.get("account_index") if isinstance(cand.get("account_index"), int) else None

        full_acc: Mapping[str, Any] | None = None
        if account_index is not None:
            full_acc = accounts_by_index.get(account_index)
        if full_acc is None and account_id is not None:
            full_acc = lookup.get(str(account_id))
            if isinstance(full_acc, Mapping):
                idx_from_acc = full_acc.get("account_index")
                if isinstance(idx_from_acc, int):
                    account_index = idx_from_acc

        if not isinstance(account_index, int):
            logger.warning("CASE_BUILD_SKIP sid=%s account_id=%s reason=no_account_index", sid, account_id)
            continue

        if not isinstance(full_acc, Mapping):
            full_acc = accounts_by_index.get(account_index)

        if not isinstance(full_acc, Mapping):
            logger.warning("CASE_BUILD_SKIP sid=%s idx=%s reason=no_full_account", sid, account_index)
            continue

        account_id_str = str(account_id) if account_id is not None else f"idx-{account_index:03d}"
        acc_dir = (accounts_dir / str(account_index)).resolve()
        acc_dir.mkdir(parents=True, exist_ok=True)

        pointers = {
            "raw": "raw_lines.json",
            "bureaus": "bureaus.json",
            "flat": "fields_flat.json",
            "tags": "tags.json",
            "summary": "summary.json",
        }

        raw_lines = list(full_acc.get("lines") or [])
        (acc_dir / pointers["raw"]).write_text(
            json.dumps(raw_lines, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        bureaus_obj = dict(full_acc.get("triad_fields") or {})
        (acc_dir / pointers["bureaus"]).write_text(
            json.dumps(bureaus_obj, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        flat_fields, _prov = build_rule_fields_from_triad(dict(full_acc))
        (acc_dir / pointers["flat"]).write_text(
            json.dumps(flat_fields, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        tags_path = acc_dir / pointers["tags"]
        if not tags_path.exists():
            tags_path.write_text("[]", encoding="utf-8")

        meta_obj = {
            "account_index": account_index,
            "heading_guess": full_acc.get("heading_guess"),
            "page_start": full_acc.get("page_start"),
            "line_start": full_acc.get("line_start"),
            "page_end": full_acc.get("page_end"),
            "line_end": full_acc.get("line_end"),
            "pointers": pointers,
        }
        if account_id is not None:
            meta_obj["account_id"] = account_id
        (acc_dir / "meta.json").write_text(
            json.dumps(meta_obj, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        summary_path = acc_dir / pointers["summary"]
        existing_summary, loaded_existing = _load_summary_json(summary_path)
        if loaded_existing:
            summary_obj = dict(existing_summary)
        else:
            summary_obj = {}

        summary_obj["sid"] = sid
        summary_obj["account_index"] = account_index
        if account_id is not None:
            summary_obj["account_id"] = account_id

        candidate_reason = cand.get("reason") if isinstance(cand.get("reason"), Mapping) else None
        candidate_primary_issue = (
            candidate_reason.get("primary_issue")
            if isinstance(candidate_reason, Mapping)
            else cand.get("primary_issue")
        )
        candidate_problem_reasons = (
            candidate_reason.get("problem_reasons")
            if isinstance(candidate_reason, Mapping)
            else cand.get("problem_reasons")
        )
        candidate_problem_tags = cand.get("problem_tags")

        coerced_tags = _coerce_list(candidate_problem_tags)
        coerced_reasons = _coerce_list(candidate_problem_reasons)

        if not loaded_existing:
            if coerced_tags is not None:
                summary_obj["problem_tags"] = coerced_tags
            else:
                summary_obj.setdefault("problem_tags", [])
            if coerced_reasons is not None:
                summary_obj["problem_reasons"] = coerced_reasons
            else:
                summary_obj.setdefault("problem_reasons", [])
            if candidate_primary_issue is not None:
                summary_obj["primary_issue"] = candidate_primary_issue
            if "confidence" in cand and cand.get("confidence") is not None:
                summary_obj["confidence"] = cand.get("confidence")
        else:
            if "problem_tags" not in summary_obj and coerced_tags is not None:
                summary_obj["problem_tags"] = coerced_tags
            if "problem_reasons" not in summary_obj and coerced_reasons is not None:
                summary_obj["problem_reasons"] = coerced_reasons
            if "primary_issue" not in summary_obj and candidate_primary_issue is not None:
                summary_obj["primary_issue"] = candidate_primary_issue

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
                merge_groups[str(account_index)] = group_id
        else:
            existing_tag = summary_obj.get("merge_tag")
            if isinstance(existing_tag, Mapping):
                group_id = existing_tag.get("group_id")
                if isinstance(group_id, str):
                    merge_groups[str(account_index)] = group_id

        summary_obj["pointers"] = pointers

        if not loaded_existing and isinstance(general_info, dict):
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

        summary_path.write_text(
            json.dumps(summary_obj, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        try:
            m.set_artifact(f"cases.accounts.{account_index}", "dir", acc_dir)
            m.set_artifact(f"cases.accounts.{account_index}", "meta", acc_dir / "meta.json")
            m.set_artifact(
                f"cases.accounts.{account_index}", "raw", acc_dir / pointers["raw"]
            )
            m.set_artifact(
                f"cases.accounts.{account_index}", "bureaus", acc_dir / pointers["bureaus"]
            )
            m.set_artifact(
                f"cases.accounts.{account_index}", "flat", acc_dir / pointers["flat"]
            )
            m.set_artifact(
                f"cases.accounts.{account_index}", "summary", summary_path
            )
            m.set_artifact(
                f"cases.accounts.{account_index}", "tags", acc_dir / pointers["tags"]
            )
        except Exception:
            pass

        written_ids.append(str(account_index))

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
        item = {
            "id": aid,
            "dir": str((accounts_dir / aid).resolve()),
        }
        try:
            aid_int = int(aid)
        except (TypeError, ValueError):
            aid_int = None
        if aid_int is not None:
            full_acc = accounts_by_index.get(aid_int)
            if isinstance(full_acc, Mapping):
                account_id_val = full_acc.get("account_id")
                if account_id_val is not None:
                    item["account_id"] = account_id_val
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
