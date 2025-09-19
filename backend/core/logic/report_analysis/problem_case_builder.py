"""Lean problem case builder for Stage-A accounts.

This module materialises a compact set of JSON artefacts for each
problematic account detected by the analyzer.  Only the fields required by
operators are persisted which keeps case folders small and guarantees that
``triad_rows`` never land on disk.
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple

from backend.pipeline.runs import RunManifest, write_breadcrumb
from backend.core.logic.report_analysis.problem_extractor import (
    build_rule_fields_from_triad,
    load_stagea_accounts_from_manifest,
)

from .keys import compute_logical_account_key

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[4]))

LEAN = os.getenv("CASES_LEAN_MODE", "1") != "0"

ALLOWED_BUREAUS_TOPLEVEL = ("transunion", "experian", "equifax")


def _candidate_manifest_paths(sid: str, root: Path | None = None) -> List[Path]:
    candidates: List[Path] = []

    env_manifest = os.getenv("REPORT_MANIFEST_PATH")
    if env_manifest:
        candidates.append(Path(env_manifest))

    runs_root_env = os.getenv("RUNS_ROOT")
    if runs_root_env:
        candidates.append(Path(runs_root_env) / sid / "manifest.json")

    if root is not None:
        candidates.append(Path(root) / "runs" / sid / "manifest.json")

    candidates.append(PROJECT_ROOT / "runs" / sid / "manifest.json")

    unique: List[Path] = []
    seen: set[Path] = set()
    for path in candidates:
        try:
            resolved = path.resolve()
        except RuntimeError:
            resolved = path
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)
    return unique


def _load_manifest_for_sid(sid: str, root: Path | None = None) -> RunManifest | None:
    for path in _candidate_manifest_paths(sid, root=root):
        if path.exists():
            try:
                return RunManifest(path).load()
            except Exception:
                logger.debug(
                    "CASE_BUILDER manifest_load_failed sid=%s path=%s", sid, path
                )
    return None


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


def _sanitize_bureau_fields(payload: Any) -> Any:
    if isinstance(payload, Mapping):
        return {key: value for key, value in payload.items() if key != "triad_rows"}
    if payload is None:
        return {}
    return payload


def _sanitize_bureaus(data: Mapping[str, Any] | None) -> Dict[str, Any]:
    cleaned: Dict[str, Any] = {}
    for bureau, payload in (data or {}).items():
        cleaned[bureau] = _sanitize_bureau_fields(payload)
    return cleaned


def _build_bureaus_payload_from_stagea(
    acc: Mapping[str, Any] | None,
) -> OrderedDict[str, Any]:
    if not isinstance(acc, Mapping):
        acc = {}

    sanitized_bureaus = _sanitize_bureaus(acc.get("triad_fields"))

    ordered: "OrderedDict[str, Any]" = OrderedDict()
    for bureau in ALLOWED_BUREAUS_TOPLEVEL:
        value = sanitized_bureaus.get(bureau)
        if value is None:
            ordered[bureau] = {}
        else:
            ordered[bureau] = value

    two_year = acc.get("two_year_payment_history") or {}
    seven_year = acc.get("seven_year_history") or {}

    ordered["two_year_payment_history"] = two_year
    ordered["seven_year_history"] = seven_year
    ordered["order"] = list(ALLOWED_BUREAUS_TOPLEVEL)

    return ordered


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


# ---------------------------------------------------------------------------
# Legacy helpers
# ---------------------------------------------------------------------------


def _make_account_id(account: Mapping[str, Any], idx: int) -> str:
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
    by_key: Dict[str, Mapping[str, Any]] = {}
    for idx, acc in enumerate(accounts, start=1):
        account_index = acc.get("account_index")
        if isinstance(account_index, int):
            by_key[str(account_index)] = acc

        acc_id = _make_account_id(acc, idx)
        by_key[acc_id] = acc

    return by_key


def _resolve_inputs_from_manifest(
    sid: str, *, root: Path | None = None
) -> tuple[Path, Path, RunManifest]:
    m = _load_manifest_for_sid(sid, root=root)
    if m is None:
        raise RuntimeError(f"Run manifest not found for sid={sid}")
    try:
        accounts_path = Path(m.get("traces.accounts_table", "accounts_json")).resolve()
        general_path = Path(m.get("traces.accounts_table", "general_json")).resolve()
    except KeyError as e:
        raise RuntimeError(
            f"Run manifest missing traces.accounts_table key for sid={sid}: {e}"
        )
    return accounts_path, general_path, m


# ---------------------------------------------------------------------------
# Lean writer implementation
# ---------------------------------------------------------------------------


def _build_problem_cases_lean(
    sid: str, candidates: List[Dict[str, Any]], *, root: Path | None = None
) -> Dict[str, Any]:
    """Build lean per-account case folders under ``runs/<sid>/cases``."""

    try:
        full_accounts = load_stagea_accounts_from_manifest(sid, root=root)
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

    manifest = _load_manifest_for_sid(sid, root=root)
    if manifest is not None:
        cases_dir = manifest.ensure_run_subdir("cases_dir", "cases")
        accounts_dir = (cases_dir / "accounts").resolve()
        accounts_dir.mkdir(parents=True, exist_ok=True)
        manifest.set_base_dir("cases_accounts_dir", accounts_dir)
        write_breadcrumb(manifest.path, cases_dir / ".manifest")
    else:
        base_root = Path(root) if root is not None else PROJECT_ROOT
        cases_dir = (base_root / "cases" / sid).resolve()
        accounts_dir = cases_dir / "accounts"
        accounts_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = cases_dir / ".manifest"
        manifest_path.write_text("missing", encoding="utf-8")

    logger.info("PROBLEM_CASES start sid=%s total=%d out=%s", sid, total, cases_dir)

    written_ids: List[str] = []
    merge_groups: Dict[str, str] = {}

    for cand in candidates:
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

        bureaus_payload = _build_bureaus_payload_from_stagea(account)
        _write_json(account_dir / pointers["bureaus"], bureaus_payload)

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

        merge_tag_v2_obj: Any = (
            cand.get("merge_tag_v2") if isinstance(cand, Mapping) else None
        )
        if merge_tag_v2_obj is None and had_existing:
            merge_tag_v2_obj = existing_summary.get("merge_tag_v2")

        if isinstance(merge_tag_v2_obj, Mapping):
            try:
                sanitized_tag_v2 = json.loads(
                    json.dumps(merge_tag_v2_obj, ensure_ascii=False)
                )
            except TypeError:
                sanitized_tag_v2 = dict(merge_tag_v2_obj)
            summary_obj["merge_tag_v2"] = sanitized_tag_v2
        elif merge_tag_v2_obj is not None:
            summary_obj["merge_tag_v2"] = merge_tag_v2_obj

        _write_json(summary_path, summary_obj)

        artifact_keys = {str(idx)}
        if account_id is not None:
            artifact_keys.add(str(account_id))

        if manifest is None:
            legacy_id = str(account_id) if account_id is not None else f"idx-{idx:03d}"
            legacy_path = accounts_dir / f"{legacy_id}.json"
            legacy_payload = dict(summary_obj)
            legacy_payload.setdefault("case_dir", str(account_dir))
            try:
                _write_json(legacy_path, legacy_payload)
            except Exception:
                logger.debug(
                    "CASE_BUILD_LEGACY_WRITE_FAILED sid=%s path=%s", sid, legacy_path
                )

        if manifest is not None:
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

    candidates_list = [c for c in candidates if isinstance(c, Mapping)]
    index_payload = {
        "sid": sid,
        "total": total,
        "problematic": len(candidates_list),
        "problematic_accounts": [c.get("account_index") for c in candidates_list],
    }
    _write_json(cases_dir / "index.json", index_payload)
    if manifest is not None:
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


# ---------------------------------------------------------------------------
# Legacy writer implementation
# ---------------------------------------------------------------------------


def _build_problem_cases_legacy(
    sid: str, candidates: List[Dict[str, Any]], *, root: Path | None = None
) -> Dict[str, Any]:
    """Materialise legacy problem case files for ``sid``."""

    acc_path, gen_path, manifest = _resolve_inputs_from_manifest(sid, root=root)
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

    general_info: Dict[str, Any] | None = None
    try:
        if gen_path and gen_path.exists():
            general_info = json.loads(gen_path.read_text(encoding="utf-8"))
    except Exception:
        general_info = None

    cases_dir = manifest.ensure_run_subdir("cases_dir", "cases")
    accounts_dir = (cases_dir / "accounts").resolve()
    accounts_dir.mkdir(parents=True, exist_ok=True)
    manifest.set_base_dir("cases_accounts_dir", accounts_dir)
    write_breadcrumb(manifest.path, cases_dir / ".manifest")
    logger.info("CASES_OUT sid=%s accounts_dir=%s", sid, accounts_dir)

    logger.info("PROBLEM_CASES start sid=%s total=%s out=%s", sid, total, cases_dir)

    written_ids: List[str] = []
    merge_groups: Dict[str, str] = {}
    for cand in candidates:
        if not isinstance(cand, Mapping):
            continue

        account_id = cand.get("account_id")
        account_index = (
            cand.get("account_index") if isinstance(cand.get("account_index"), int) else None
        )

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
            logger.warning(
                "CASE_BUILD_SKIP sid=%s account_id=%s reason=no_account_index", sid, account_id
            )
            continue

        if not isinstance(full_acc, Mapping):
            full_acc = accounts_by_index.get(account_index)

        if not isinstance(full_acc, Mapping):
            logger.warning(
                "CASE_BUILD_SKIP sid=%s idx=%s reason=no_full_account", sid, account_index
            )
            continue

        account_dir = (accounts_dir / str(account_index)).resolve()
        account_dir.mkdir(parents=True, exist_ok=True)

        pointers = {
            "raw": "raw_lines.json",
            "bureaus": "bureaus.json",
            "flat": "fields_flat.json",
            "tags": "tags.json",
            "summary": "summary.json",
        }

        raw_lines = list(full_acc.get("lines") or [])
        _write_json(account_dir / pointers["raw"], raw_lines)

        bureaus_obj = _build_bureaus_payload_from_stagea(full_acc)
        _write_json(account_dir / pointers["bureaus"], bureaus_obj)

        flat_fields, _prov = build_rule_fields_from_triad(dict(full_acc))
        _write_json(account_dir / pointers["flat"], flat_fields)

        tags_path = account_dir / pointers["tags"]
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
        _write_json(account_dir / "meta.json", meta_obj)

        summary_path = account_dir / pointers["summary"]
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

        merge_tag_v2 = cand.get("merge_tag_v2")
        if isinstance(merge_tag_v2, Mapping):
            try:
                merge_tag_v2_obj = json.loads(
                    json.dumps(merge_tag_v2, ensure_ascii=False)
                )
            except TypeError:
                merge_tag_v2_obj = dict(merge_tag_v2)
            summary_obj["merge_tag_v2"] = merge_tag_v2_obj
        else:
            existing_v2 = summary_obj.get("merge_tag_v2")
            if existing_v2 is None:
                legacy_v2 = existing_summary.get("merge_tag_v2")
                if isinstance(legacy_v2, Mapping):
                    try:
                        legacy_v2_obj = json.loads(
                            json.dumps(legacy_v2, ensure_ascii=False)
                        )
                    except TypeError:
                        legacy_v2_obj = dict(legacy_v2)
                    summary_obj["merge_tag_v2"] = legacy_v2_obj

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

        _write_json(summary_path, summary_obj)

        try:
            manifest.set_artifact(f"cases.accounts.{account_index}", "dir", account_dir)
            manifest.set_artifact(
                f"cases.accounts.{account_index}", "meta", account_dir / "meta.json"
            )
            manifest.set_artifact(
                f"cases.accounts.{account_index}", "raw", account_dir / pointers["raw"]
            )
            manifest.set_artifact(
                f"cases.accounts.{account_index}", "bureaus", account_dir / pointers["bureaus"]
            )
            manifest.set_artifact(
                f"cases.accounts.{account_index}", "flat", account_dir / pointers["flat"]
            )
            manifest.set_artifact(
                f"cases.accounts.{account_index}", "summary", summary_path
            )
            manifest.set_artifact(
                f"cases.accounts.{account_index}", "tags", account_dir / pointers["tags"]
            )
        except Exception:
            pass

        written_ids.append(str(account_index))

    cand_list = list(candidates)
    index_data = {
        "sid": sid,
        "total": total,
        "problematic": len(cand_list),
        "problematic_accounts": [c.get("account_id") for c in cand_list],
    }
    _write_json(cases_dir / "index.json", index_data)
    manifest.set_artifact("cases", "accounts_index", accounts_dir / "index.json")
    manifest.set_artifact("cases", "problematic_ids", cases_dir / "index.json")

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
    _write_json(accounts_index_path, acc_index)
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
        "cases": {"count": len(written_ids), "dir": str(accounts_dir)},
    }


# ---------------------------------------------------------------------------
# Public entry point with feature flag
# ---------------------------------------------------------------------------


def build_problem_cases(
    sid: str,
    candidates: List[Dict[str, Any]],
    *,
    root: Path | None = None,
) -> Dict[str, Any]:
    if candidates is None:  # pragma: no cover - defensive guard
        raise TypeError("candidates must not be None")

    cand_list = list(candidates)

    if not LEAN:
        return _build_problem_cases_legacy(sid, candidates=cand_list, root=root)
    return _build_problem_cases_lean(sid, candidates=cand_list, root=root)


__all__ = ["build_problem_cases"]
