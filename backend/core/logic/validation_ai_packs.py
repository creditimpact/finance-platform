"""Helpers for building validation AI adjudication packs."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from backend.core.ai.paths import (
    ValidationAccountPaths,
    ensure_validation_account_paths,
    ensure_validation_paths,
)
from backend.pipeline.runs import RunManifest

log = logging.getLogger(__name__)


def _normalize_indices(indices: Iterable[int | str]) -> list[int]:
    normalized: set[int] = set()
    for idx in indices:
        try:
            normalized.add(int(str(idx)))
        except Exception:
            continue
    return sorted(normalized)


def build_validation_ai_packs_for_accounts(
    sid: str,
    *,
    account_indices: Sequence[int | str],
    runs_root: Path | str | None = None,
) -> None:
    """Trigger validation AI pack building for the provided account indices.

    The builder currently ensures the filesystem scaffold for validation AI
    packs exists so subsequent stages can populate payloads and prompts.
    """

    normalized_indices = _normalize_indices(account_indices)
    if not normalized_indices:
        return

    runs_root_path = Path(runs_root) if runs_root is not None else Path("runs")
    validation_paths = ensure_validation_paths(runs_root_path, sid, create=True)

    created: list[ValidationAccountPaths] = []
    accounts_root = runs_root_path / sid / "cases" / "accounts"

    for idx in normalized_indices:
        account_paths = ensure_validation_account_paths(
            validation_paths, idx, create=True
        )
        _ensure_placeholder_files(account_paths)

        summary = _load_summary(accounts_root, idx)
        weak_items = _collect_weak_items(summary)
        pack_payload = {"weak_items": weak_items}
        _write_pack(account_paths.pack_file, pack_payload)

        created.append(account_paths)

    manifest_path = runs_root_path / sid / "manifest.json"
    manifest = RunManifest.load_or_create(manifest_path, sid)
    if created:
        last_account = created[-1]
        index_path = validation_paths.base / "index.json"
        manifest.upsert_validation_packs_dir(
            validation_paths.base,
            account_dir=last_account.base,
            results_dir=last_account.results_dir,
            index_file=index_path,
        )
    else:
        manifest.upsert_validation_packs_dir(validation_paths.base)

    log.info(
        "VALIDATION_AI_PACKS_INITIALIZED sid=%s base=%s accounts=%s",
        sid,
        validation_paths.base,
        ",".join(str(path.base.name) for path in created),
    )


def _ensure_placeholder_files(paths: ValidationAccountPaths) -> None:
    """Create empty scaffold files for a validation pack if they are missing."""

    _ensure_file(paths.pack_file, "{}\n")
    _ensure_file(paths.prompt_file, "")
    _ensure_file(paths.model_results_file, "{}\n")


def _ensure_file(path: Path, default_contents: str) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(default_contents, encoding="utf-8")


def _load_summary(accounts_root: Path, account_idx: int) -> Mapping[str, Any] | None:
    """Return the parsed summary.json payload for ``account_idx`` if present."""

    summary_path = accounts_root / str(account_idx) / "summary.json"
    try:
        raw_text = summary_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError:
        log.warning(
            "VALIDATION_SUMMARY_READ_FAILED account=%s path=%s",
            account_idx,
            summary_path,
            exc_info=True,
        )
        return None

    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError:
        log.warning(
            "VALIDATION_SUMMARY_INVALID_JSON account=%s path=%s",
            account_idx,
            summary_path,
            exc_info=True,
        )
        return None

    if not isinstance(payload, Mapping):
        log.warning(
            "VALIDATION_SUMMARY_INVALID_TYPE account=%s path=%s type=%s",
            account_idx,
            summary_path,
            type(payload).__name__,
        )
        return None

    return payload


def _collect_weak_items(summary: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    """Extract validation requirements that require AI adjudication."""

    if not isinstance(summary, Mapping):
        return []

    validation = summary.get("validation_requirements")
    if not isinstance(validation, Mapping):
        return []

    requirements = validation.get("requirements")
    if not isinstance(requirements, Sequence):
        return []

    field_consistency = validation.get("field_consistency")
    if isinstance(field_consistency, Mapping):
        consistency_map: Mapping[str, Any] = field_consistency
    else:
        consistency_map = {}

    weak_items: list[dict[str, Any]] = []

    for entry in requirements:
        if not isinstance(entry, Mapping):
            continue

        if not entry.get("ai_needed"):
            continue

        raw_field = entry.get("field")
        if raw_field is None:
            continue

        field = str(raw_field)

        documents = entry.get("documents")
        if isinstance(documents, Sequence) and not isinstance(
            documents, (str, bytes, bytearray)
        ):
            documents_list = [str(doc) for doc in documents]
        elif documents is None:
            documents_list = []
        else:
            documents_list = [str(documents)]

        item: dict[str, Any] = {
            "field": field,
            "category": entry.get("category"),
            "min_days": entry.get("min_days"),
            "documents": documents_list,
        }

        consistency_details = consistency_map.get(field)
        if isinstance(consistency_details, Mapping):
            item["consensus"] = consistency_details.get("consensus")

            disagreeing = consistency_details.get("disagreeing_bureaus")
            if isinstance(disagreeing, Sequence) and not isinstance(
                disagreeing, (str, bytes, bytearray)
            ):
                item["disagreeing_bureaus"] = sorted(str(b) for b in disagreeing)
            else:
                item["disagreeing_bureaus"] = []

            missing = consistency_details.get("missing_bureaus")
            if isinstance(missing, Sequence) and not isinstance(
                missing, (str, bytes, bytearray)
            ):
                item["missing_bureaus"] = sorted(str(b) for b in missing)
            else:
                item["missing_bureaus"] = []

            raw_values = consistency_details.get("raw")
            raw_map = raw_values if isinstance(raw_values, Mapping) else {}

            normalized_values = consistency_details.get("normalized")
            normalized_map = (
                normalized_values if isinstance(normalized_values, Mapping) else {}
            )

            values: dict[str, dict[str, Any]] = {}
            for bureau in ("transunion", "experian", "equifax"):
                values[bureau] = {
                    "raw": raw_map.get(bureau),
                    "normalized": normalized_map.get(bureau),
                }

            item["values"] = values
        else:
            item["consensus"] = None
            item["disagreeing_bureaus"] = []
            item["missing_bureaus"] = []
            item["values"] = {
                bureau: {"raw": None, "normalized": None}
                for bureau in ("transunion", "experian", "equifax")
            }

        weak_items.append(item)

    return weak_items


def _write_pack(path: Path, payload: Mapping[str, Any]) -> None:
    """Write ``payload`` to ``path`` as JSON."""

    try:
        serialized = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    except TypeError:
        log.exception("VALIDATION_PACK_SERIALIZE_FAILED path=%s", path)
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(serialized + "\n", encoding="utf-8")

