"""Helpers for building validation AI adjudication packs."""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
import textwrap
from typing import Any, Iterable, Mapping, Sequence

import yaml

from backend.core.ai.paths import (
    ValidationAccountPaths,
    ensure_validation_account_paths,
    ensure_validation_paths,
)
from backend.pipeline.runs import RunManifest
from backend.core.logic.utils.json_utils import parse_json

log = logging.getLogger(__name__)

_DEFAULT_MODEL = "gpt-4o-mini"


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
    ai_client: Any | None = None,
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

    packs_config = _load_packs_config(validation_paths.base)
    model_name = _select_model_name(packs_config)

    ai_client = ai_client if ai_client is not None else _build_ai_client()

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

        if weak_items:
            prompt_text = _render_prompt(sid, idx, weak_items)
            _write_prompt(account_paths.prompt_file, prompt_text)
        else:
            prompt_text = ""
            _write_prompt(account_paths.prompt_file, prompt_text)

        result_payload = _run_model_inference(
            ai_client,
            prompt_text,
            model_name,
            sid=sid,
            account_idx=idx,
            has_weak_items=bool(weak_items),
        )
        _write_model_results(account_paths.model_results_file, result_payload)

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


def _render_prompt(sid: str, account_idx: int, weak_items: Sequence[Mapping[str, Any]]) -> str:
    schema_block = textwrap.dedent(
        """{
  \"sid\": \"<copy the SID from the user section>\",
  \"account_index\": <copy the account index as an integer>,
  \"decisions\": [
    {
      \"field\": \"<field name>\",
      \"decision\": \"STRONG\" | \"NO_CLAIM\"
    }
  ]
}"""
    )

    system_lines = textwrap.dedent(
        """SYSTEM:
You are an adjudication assistant reviewing credit report inconsistencies.
Evaluate each weak field independently and decide whether the consumer has a strong claim.
Follow these rules:
1. Base decisions only on the provided data for each field.
2. Return the decisions in the same order the fields are provided.
3. Use decision value \"STRONG\" when the evidence indicates a strong inconsistency; otherwise use \"NO_CLAIM\".
Return a STRICT JSON object matching this schema (no extra keys, commentary, or trailing characters):
"""
    )

    system_lines += textwrap.indent(schema_block, "  ")
    system_lines += textwrap.dedent(
        """

Set \"sid\" to the provided SID and \"account_index\" to the provided account index.
Do not include any explanation outside the JSON response.
"""
    )

    weak_fields_json = json.dumps(
        list(weak_items), indent=2, sort_keys=True, ensure_ascii=False
    )

    prompt = (
        f"{system_lines}\n\n"
        "USER:\n"
        f"SID: {sid}\n"
        f"ACCOUNT_INDEX: {account_idx}\n"
        "WEAK_FIELDS:\n"
        f"{weak_fields_json}\n"
    )

    return prompt


def _write_prompt(path: Path, prompt: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(prompt, encoding="utf-8")


def _write_model_results(path: Path, payload: Mapping[str, Any]) -> None:
    try:
        serialized = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    except TypeError:
        log.exception("VALIDATION_AI_RESULTS_SERIALIZE_FAILED path=%s", path)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(serialized + "\n", encoding="utf-8")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def _load_packs_config(base_dir: Path) -> Mapping[str, Any]:
    config_path = base_dir / "ai_packs_config.yml"
    if not config_path.exists():
        return {}
    try:
        data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except Exception:
        log.warning(
            "VALIDATION_AI_CONFIG_LOAD_FAILED path=%s", config_path, exc_info=True
        )
        return {}
    if isinstance(data, Mapping):
        return data
    log.warning(
        "VALIDATION_AI_CONFIG_INVALID path=%s type=%s",
        config_path,
        type(data).__name__,
    )
    return {}


def _select_model_name(config: Mapping[str, Any]) -> str:
    raw_model = config.get("model") if isinstance(config, Mapping) else None
    if isinstance(raw_model, str) and raw_model.strip():
        return raw_model.strip()
    return _DEFAULT_MODEL


def _build_ai_client() -> Any | None:
    try:
        from backend.core.services.ai_client import get_ai_client

        return get_ai_client()
    except Exception:
        log.warning("VALIDATION_AI_CLIENT_UNAVAILABLE", exc_info=True)
        return None


def _extract_response_text(response: Any) -> str | None:
    if response is None:
        return None

    for attr in ("output_text", "text"):
        value = getattr(response, attr, None)
        if isinstance(value, str) and value.strip():
            return value

    output = getattr(response, "output", None)
    if isinstance(output, list) and output:
        first = output[0]
        content = getattr(first, "content", None)
        if isinstance(content, list) and content:
            first_content = content[0]
            text_val = getattr(first_content, "text", None)
            if isinstance(text_val, str) and text_val.strip():
                return text_val
            if isinstance(first_content, Mapping):
                text_val = first_content.get("text")
                if isinstance(text_val, str) and text_val.strip():
                    return text_val

    if isinstance(response, Mapping):
        text_val = response.get("output_text") or response.get("text")
        if isinstance(text_val, str) and text_val.strip():
            return text_val

    return None


def _run_model_inference(
    ai_client: Any | None,
    prompt: str,
    model: str,
    *,
    sid: str,
    account_idx: int | str,
    has_weak_items: bool,
) -> dict[str, Any]:
    timestamp = _utc_now()

    if not has_weak_items:
        return {
            "status": "skipped",
            "reason": "no_weak_items",
            "model": model,
            "timestamp": timestamp,
            "duration_ms": 0,
        }

    if ai_client is None:
        log.warning(
            "VALIDATION_AI_CLIENT_MISSING sid=%s account=%s", sid, account_idx
        )
        return {
            "status": "skipped",
            "reason": "no_client",
            "model": model,
            "timestamp": timestamp,
            "duration_ms": 0,
        }

    started = time.perf_counter()
    try:
        response = ai_client.response_json(
            prompt=prompt,
            model=model,
            response_format={"type": "json_object"},
        )
        duration_ms = int((time.perf_counter() - started) * 1000)
    except Exception as exc:  # pragma: no cover - defensive logging
        duration_ms = int((time.perf_counter() - started) * 1000)
        log.warning(
            "VALIDATION_AI_CALL_FAILED sid=%s account=%s error=%s",
            sid,
            account_idx,
            exc,
        )
        return {
            "status": "error",
            "reason": exc.__class__.__name__,
            "model": model,
            "timestamp": timestamp,
            "duration_ms": duration_ms,
        }

    raw_text = _extract_response_text(response)
    result: dict[str, Any] = {
        "status": "ok",
        "model": model,
        "timestamp": timestamp,
        "duration_ms": duration_ms,
    }

    if raw_text is None:
        result.update({"status": "error", "reason": "empty_response"})
        return result

    result["raw"] = raw_text
    parsed, error_reason = parse_json(raw_text)
    result["response"] = parsed
    if error_reason:
        result["status"] = "error"
        result["reason"] = error_reason

    return result

