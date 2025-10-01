"""Helpers for building validation AI adjudication packs."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
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
_CONFIG_PATH = Path(__file__).with_name("ai_packs_config.yml")
_DEFAULT_RETRY_BACKOFF = (1.0, 3.0, 10.0)


@dataclass(frozen=True)
class ValidationPacksConfig:
    """Configuration for validation AI pack generation."""

    enable_write: bool = True
    enable_infer: bool = True
    model: str = _DEFAULT_MODEL
    weak_limit: int = 0
    max_attempts: int = 3
    backoff_seconds: tuple[float, ...] = _DEFAULT_RETRY_BACKOFF


def _load_yaml_mapping(path: Path) -> Mapping[str, Any]:
    try:
        raw_text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    except OSError:
        log.warning("VALIDATION_AI_CONFIG_READ_FAILED path=%s", path, exc_info=True)
        return {}

    try:
        loaded = yaml.safe_load(raw_text) or {}
    except Exception:
        log.warning("VALIDATION_AI_CONFIG_PARSE_FAILED path=%s", path, exc_info=True)
        return {}

    if isinstance(loaded, Mapping):
        return loaded

    log.warning(
        "VALIDATION_AI_CONFIG_TYPE_INVALID path=%s type=%s",
        path,
        type(loaded).__name__,
    )
    return {}


@lru_cache(maxsize=1)
def _load_global_config_section() -> Mapping[str, Any]:
    data = _load_yaml_mapping(_CONFIG_PATH)
    section = data.get("validation_packs") if isinstance(data, Mapping) else None
    if isinstance(section, Mapping):
        return dict(section)
    return dict(data) if isinstance(data, Mapping) else {}


def _load_local_config_section(base_dir: Path) -> Mapping[str, Any]:
    config_path = Path(base_dir) / "ai_packs_config.yml"
    data = _load_yaml_mapping(config_path)
    section = data.get("validation_packs") if isinstance(data, Mapping) else None
    if isinstance(section, Mapping):
        return dict(section)
    return dict(data) if isinstance(data, Mapping) else {}


def _coerce_bool(raw: Any, default: bool) -> bool:
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        lowered = raw.strip().lower()
        if lowered in {"1", "true", "yes", "y", "on"}:
            return True
        if lowered in {"0", "false", "no", "n", "off"}:
            return False
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        return bool(raw)
    return default


def _coerce_int(raw: Any, default: int, *, minimum: int | None = None) -> int:
    try:
        value = int(raw)
    except (TypeError, ValueError):
        value = int(default)
    if minimum is not None and value < minimum:
        return minimum
    return value


def _coerce_str(raw: Any, default: str) -> str:
    if isinstance(raw, str):
        text = raw.strip()
        return text or default
    if raw is None:
        return default
    return str(raw)


def _coerce_backoff(raw: Any) -> tuple[int, tuple[float, ...]]:
    attempts = 3
    schedule: tuple[float, ...] = _DEFAULT_RETRY_BACKOFF

    if isinstance(raw, Mapping):
        raw_attempts = raw.get("max_attempts") or raw.get("attempts")
        if raw_attempts is not None:
            attempts = _coerce_int(raw_attempts, attempts, minimum=1)

        backoff_value = (
            raw.get("backoff_seconds")
            or raw.get("backoff")
            or raw.get("delays")
            or raw.get("schedule")
        )
        if isinstance(backoff_value, Sequence) and not isinstance(
            backoff_value, (str, bytes, bytearray)
        ):
            parsed = _coerce_float_sequence(backoff_value)
            if parsed:
                schedule = parsed
        elif backoff_value is not None:
            try:
                single = float(backoff_value)
            except (TypeError, ValueError):
                single = None
            if single is not None:
                schedule = (max(0.0, single),)

    elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
        parsed = _coerce_float_sequence(raw)
        if parsed:
            schedule = parsed
            attempts = max(attempts, len(schedule) + 1)
    elif raw is not None:
        try:
            single_val = float(raw)
        except (TypeError, ValueError):
            single_val = None
        if single_val is not None:
            schedule = (max(0.0, single_val),)
            attempts = max(attempts, 2)

    if attempts < 1:
        attempts = 1

    if not schedule:
        schedule = _DEFAULT_RETRY_BACKOFF

    return attempts, schedule


def _coerce_float_sequence(raw: Sequence[Any]) -> tuple[float, ...]:
    values: list[float] = []
    for entry in raw:
        if entry is None:
            continue
        try:
            val = float(entry)
        except (TypeError, ValueError):
            continue
        values.append(max(0.0, val))
    return tuple(values)


def _coerce_validation_config(raw: Mapping[str, Any]) -> ValidationPacksConfig:
    enable_write = _coerce_bool(raw.get("enable_write"), True)
    enable_infer = _coerce_bool(raw.get("enable_infer"), True)
    model = _coerce_str(raw.get("model"), _DEFAULT_MODEL)
    weak_limit = _coerce_int(raw.get("weak_limit"), 0, minimum=0)

    attempts, backoff_schedule = _coerce_backoff(raw.get("retry"))

    return ValidationPacksConfig(
        enable_write=enable_write,
        enable_infer=enable_infer,
        model=model,
        weak_limit=weak_limit,
        max_attempts=attempts,
        backoff_seconds=backoff_schedule,
    )


def load_validation_packs_config(
    base_dir: Path | str | None = None,
) -> ValidationPacksConfig:
    """Return the effective validation packs configuration."""

    base_path = Path(base_dir) if base_dir is not None else None

    merged: dict[str, Any] = {}
    merged.update(_load_global_config_section())
    if base_path is not None:
        merged.update(_load_local_config_section(base_path))

    return _coerce_validation_config(merged)


def load_validation_packs_config_for_run(
    sid: str,
    *,
    runs_root: Path | str | None = None,
) -> ValidationPacksConfig:
    """Convenience wrapper to read config for ``sid`` without touching disk."""

    root_path = Path(runs_root) if runs_root is not None else Path("runs")
    base_dir = root_path / sid / "ai_packs" / "validation"
    return load_validation_packs_config(base_dir)


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
    base_dir = runs_root_path / sid / "ai_packs" / "validation"
    packs_config = load_validation_packs_config(base_dir)

    if not packs_config.enable_write:
        log.info(
            "VALIDATION_AI_PACKS_DISABLED sid=%s reason=write_disabled base=%s",
            sid,
            base_dir,
        )
        return

    validation_paths = ensure_validation_paths(runs_root_path, sid, create=True)

    model_name = packs_config.model

    if ai_client is None and packs_config.enable_infer:
        ai_client = _build_ai_client()

    if not packs_config.enable_infer:
        ai_client = None

    created: list[ValidationAccountPaths] = []
    accounts_root = runs_root_path / sid / "cases" / "accounts"

    for idx in normalized_indices:
        account_paths = ensure_validation_account_paths(
            validation_paths, idx, create=True
        )
        _ensure_placeholder_files(account_paths)

        summary = _load_summary(accounts_root, idx)
        weak_items = _collect_weak_items(summary)
        if packs_config.weak_limit > 0:
            weak_items = weak_items[: packs_config.weak_limit]
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
            config=packs_config,
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
    config: ValidationPacksConfig,
) -> dict[str, Any]:
    timestamp = _utc_now()

    if not has_weak_items:
        return {
            "status": "skipped",
            "reason": "no_weak_items",
            "model": model,
            "timestamp": timestamp,
            "duration_ms": 0,
            "attempts": 0,
        }

    if not config.enable_infer:
        return {
            "status": "skipped",
            "reason": "inference_disabled",
            "model": model,
            "timestamp": timestamp,
            "duration_ms": 0,
            "attempts": 0,
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
            "attempts": 0,
        }

    attempts = 0
    total_duration_ms = 0
    response: Any | None = None
    last_error: Exception | None = None

    max_attempts = max(1, config.max_attempts)

    while attempts < max_attempts:
        attempts += 1
        started = time.perf_counter()
        try:
            response = ai_client.response_json(
                prompt=prompt,
                model=model,
                response_format={"type": "json_object"},
            )
            total_duration_ms += int((time.perf_counter() - started) * 1000)
            last_error = None
            break
        except Exception as exc:  # pragma: no cover - defensive logging
            total_duration_ms += int((time.perf_counter() - started) * 1000)
            last_error = exc
            log.warning(
                "VALIDATION_AI_CALL_FAILED sid=%s account=%s attempt=%s error=%s",
                sid,
                account_idx,
                attempts,
                exc,
            )
            if attempts >= max_attempts:
                break
            backoff_idx = min(attempts - 1, len(config.backoff_seconds) - 1)
            delay = config.backoff_seconds[backoff_idx]
            if delay > 0:
                time.sleep(delay)

    if last_error is not None or response is None:
        reason = "unknown"
        if last_error is not None:
            reason = last_error.__class__.__name__
        return {
            "status": "error",
            "reason": reason,
            "model": model,
            "timestamp": timestamp,
            "duration_ms": total_duration_ms,
            "attempts": attempts,
        }

    raw_text = _extract_response_text(response)
    result: dict[str, Any] = {
        "status": "ok",
        "model": model,
        "timestamp": timestamp,
        "duration_ms": total_duration_ms,
        "attempts": attempts,
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

