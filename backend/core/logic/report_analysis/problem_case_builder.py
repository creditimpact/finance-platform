"""Lean problem case builder for Stage-A accounts.

This module materialises a compact set of JSON artefacts for each
problematic account detected by the analyzer.  Only the fields required by
operators are persisted which keeps case folders small and guarantees that
``triad_rows`` never land on disk.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

from backend.pipeline.runs import RunManifest, write_breadcrumb
from backend.core.logic.report_analysis.problem_extractor import (
    build_rule_fields_from_triad,
    load_stagea_accounts_from_manifest,
)

logger = logging.getLogger(__name__)


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_summary_json(path: Path) -> Tuple[Dict[str, Any], bool]:
    if not path.exists():
        return {}, False

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}, False

    if isinstance(data, dict):
        return data, True
    return {}, False


def _coerce_list(value: Any) -> List[Any] | None:
    if value is None:
        return None
    if isinstance(value, list):
        return list(value)
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _sanitize_bureaus(data: Mapping[str, Any] | None) -> Dict[str, Any]:
    cleaned: Dict[str, Any] = {}
    for bureau, payload in (data or {}).items():
        if isinstance(payload, Mapping):
            cleaned[bureau] = {
                key: value
                for key, value in payload.items()
                if key != "triad_rows"
            }
        else:
            cleaned[bureau] = payload
    return cleaned


def _extract_candidate_reason(cand: Mapping[str, Any]) -> Tuple[Any, Any, Any]:
    reason = cand.get("reason") if isinstance(cand, Mapping) else None

    primary_issue = None
    problem_reasons = None
    problem_tags = None

    if isinstance(reason, Mapping):
        primary_issue = reason.get("primary_issue")
        problem_reasons = reason.get("problem_reasons")
        problem_tags = reason.get("problem_tags")

    if primary_issue is None:
        primary_issue = cand.get("primary_issue")
    if problem_reasons is None:
        problem_reasons = cand.get("problem_reasons")
    if problem_tags is None:
        problem_tags = cand.get("problem_tags")

    return primary_issue, problem_reasons, problem_tags


def build_problem_cases(
    sid: str, candidates: List[Dict[str, Any]] | None = None
) -> Dict[str, Any]:
    """Build lean per-account case folders under ``runs/<sid>/cases``."""

    try:
        full_accounts = load_stagea_accounts_from_manifest(sid)
    except Exception as exc:
        raise RuntimeError(f"Failed to load Stage-A accounts for sid={sid}") from exc

    accounts_by_index: Dict[int, Dict[str, Any]] = {}
    for account in full_accounts:
        try:
            idx = int(account.get("account_index"))
        except Exception:
            continue
        accounts_by_index[idx] = account

    total = len(full_accounts)

    manifest = RunManifest.for_sid(sid)
    cases_dir = manifest.ensure_run_subdir("cases_dir", "cases")
    accounts_dir = (cases_dir / "accounts").resolve()
    accounts_dir.mkdir(parents=True, exist_ok=True)
    manifest.set_base_dir("cases_accounts_dir", accounts_dir)
    write_breadcrumb(manifest.path, cases_dir / ".manifest")

    logger.info("PROBLEM_CASES start sid=%s total=%d out=%s", sid, total, cases_dir)

    written_ids: List[str] = []
    merge_groups: Dict[str, str] = {}

    for cand in candidates or []:
        if not isinstance(cand, Mapping):
            continue

        idx_val = cand.get("account_index")
        try:
            idx = int(idx_val)
        except Exception:
            logger.warning(
                "CASE_BUILD_SKIP sid=%s reason=no_account_index cand=%s", sid, cand
            )
            continue

        account = accounts_by_index.get(idx)
        if not isinstance(account, Mapping):
            logger.warning(
                "CASE_BUILD_SKIP sid=%s idx=%s reason=no_full_account", sid, idx
            )
            continue

        account_dir = accounts_dir / str(idx)
        account_dir.mkdir(parents=True, exist_ok=True)

        pointers = {
            "raw": "raw_lines.json",
            "bureaus": "bureaus.json",
            "flat": "fields_flat.json",
            "tags": "tags.json",
            "summary": "summary.json",
        }

        raw_lines = list(account.get("lines") or [])
        _write_json(account_dir / pointers["raw"], raw_lines)

        bureaus = _sanitize_bureaus(account.get("triad_fields"))
        _write_json(account_dir / pointers["bureaus"], bureaus)

        flat_fields, _provenance = build_rule_fields_from_triad(dict(account))
        _write_json(account_dir / pointers["flat"], flat_fields)

        tags_path = account_dir / pointers["tags"]
        if not tags_path.exists():
            tags_path.write_text("[]", encoding="utf-8")

        meta = {
            "account_index": idx,
            "heading_guess": account.get("heading_guess"),
            "page_start": account.get("page_start"),
            "line_start": account.get("line_start"),
            "page_end": account.get("page_end"),
            "line_end": account.get("line_end"),
            "pointers": pointers,
        }
        account_id = cand.get("account_id") or account.get("account_id")
        if account_id is not None:
            meta["account_id"] = account_id
        _write_json(account_dir / "meta.json", meta)

        summary_path = account_dir / pointers["summary"]
        existing_summary, had_existing = _load_summary_json(summary_path)

        summary_obj: Dict[str, Any] = {
            "account_index": idx,
            "pointers": pointers,
        }
        if account_id is not None:
            summary_obj["account_id"] = account_id

        if had_existing:
            for key in ("problem_reasons", "problem_tags", "primary_issue"):
                if key in existing_summary:
                    summary_obj[key] = existing_summary[key]
        else:
            primary_issue, problem_reasons, problem_tags = _extract_candidate_reason(cand)
            reasons_list = _coerce_list(problem_reasons) or []
            tags_list = _coerce_list(problem_tags)

            summary_obj["problem_reasons"] = reasons_list
            if tags_list is not None:
                summary_obj["problem_tags"] = tags_list
            else:
                summary_obj["problem_tags"] = []
            if primary_issue is not None:
                summary_obj["primary_issue"] = primary_issue

        merge_tag_obj: Any = cand.get("merge_tag") if isinstance(cand, Mapping) else None
        if merge_tag_obj is None and had_existing:
            merge_tag_obj = existing_summary.get("merge_tag")

        if isinstance(merge_tag_obj, Mapping):
            try:
                sanitized_tag = json.loads(
                    json.dumps(merge_tag_obj, ensure_ascii=False)
                )
            except TypeError:
                sanitized_tag = dict(merge_tag_obj)
            summary_obj["merge_tag"] = sanitized_tag
            group_id = sanitized_tag.get("group_id")
            if isinstance(group_id, str):
                merge_groups[str(idx)] = group_id
        elif merge_tag_obj is not None:
            summary_obj["merge_tag"] = merge_tag_obj

        _write_json(summary_path, summary_obj)

        artifact_keys = {str(idx)}
        if account_id is not None:
            artifact_keys.add(str(account_id))

        try:
            for key in artifact_keys:
                manifest.set_artifact(f"cases.accounts.{key}", "dir", account_dir)
                manifest.set_artifact(
                    f"cases.accounts.{key}", "meta", account_dir / "meta.json"
                )
                manifest.set_artifact(
                    f"cases.accounts.{key}", "raw", account_dir / pointers["raw"]
                )
                manifest.set_artifact(
                    f"cases.accounts.{key}", "bureaus", account_dir / pointers["bureaus"]
                )
                manifest.set_artifact(
                    f"cases.accounts.{key}", "flat", account_dir / pointers["flat"]
                )
                manifest.set_artifact(
                    f"cases.accounts.{key}", "summary", summary_path
                )
                manifest.set_artifact(
                    f"cases.accounts.{key}", "tags", account_dir / pointers["tags"]
                )
        except Exception:
            pass

        written_ids.append(str(idx))

    candidates_list = [c for c in candidates or [] if isinstance(c, Mapping)]
    index_payload = {
        "sid": sid,
        "total": total,
        "problematic": len(candidates_list),
        "problematic_accounts": [c.get("account_index") for c in candidates_list],
    }
    _write_json(cases_dir / "index.json", index_payload)
    manifest.set_artifact("cases", "accounts_index", accounts_dir / "index.json")
    manifest.set_artifact("cases", "problematic_ids", cases_dir / "index.json")

    accounts_index = {
        "sid": sid,
        "count": len(written_ids),
        "ids": written_ids,
        "items": [],
    }
    for aid in written_ids:
        item: Dict[str, Any] = {
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
                acc_id_val = full_acc.get("account_id")
                if acc_id_val is not None:
                    item["account_id"] = acc_id_val
        group_id = merge_groups.get(aid)
        if group_id is not None:
            item["merge_group_id"] = group_id
        accounts_index["items"].append(item)

    accounts_index_path = accounts_dir / "index.json"
    _write_json(accounts_index_path, accounts_index)
    logger.info(
        "CASES_INDEX sid=%s file=%s count=%d", sid, accounts_index_path, len(written_ids)
    )

    logger.info(
        "PROBLEM_CASES done sid=%s total=%d problematic=%d out=%s",
        sid,
        total,
        len(candidates_list),
        cases_dir,
    )

    return {
        "sid": sid,
        "total": total,
        "problematic": len(candidates_list),
        "out": str(cases_dir),
        "cases": {"count": len(written_ids), "dir": str(accounts_dir)},
    }


__all__ = ["build_problem_cases"]
