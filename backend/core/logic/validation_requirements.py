"""Build and persist validation requirements derived from bureau mismatches."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence

import yaml

from backend.core.io.json_io import _atomic_write_json
from backend.core.io.tags import read_tags, write_tags_atomic
from backend.core.logic.consistency import compute_field_consistency
from backend.core.logic.summary_compact import compact_merge_sections

logger = logging.getLogger(__name__)

__all__ = [
    "ValidationRule",
    "ValidationConfig",
    "load_validation_config",
    "build_validation_requirements",
    "build_summary_payload",
    "apply_validation_summary",
    "sync_validation_tag",
    "build_validation_requirements_for_account",
]


_VALIDATION_TAG_KIND = "validation_required"
_CONFIG_PATH = Path(__file__).with_name("validation_config.yml")


@dataclass(frozen=True)
class ValidationRule:
    """Metadata describing the validation needed for a field."""

    category: str
    min_days: int
    documents: tuple[str, ...]
    points: int
    strength: str
    ai_needed: bool


@dataclass(frozen=True)
class ValidationConfig:
    defaults: ValidationRule
    fields: Mapping[str, ValidationRule]


def _coerce_documents(raw: Any, fallback: Sequence[str]) -> tuple[str, ...]:
    if isinstance(raw, (str, bytes)):
        return tuple(fallback)
    if isinstance(raw, Iterable):
        collected: List[str] = []
        for entry in raw:
            if entry is None:
                continue
            text = str(entry).strip()
            if text:
                collected.append(text)
        if collected:
            return tuple(collected)
    return tuple(fallback)


def _coerce_min_days(raw: Any, fallback: int) -> int:
    try:
        return int(raw)
    except (TypeError, ValueError):
        return int(fallback)


def _coerce_category(raw: Any, fallback: str) -> str:
    if raw is None:
        return str(fallback)
    text = str(raw).strip()
    return text or str(fallback)


def _coerce_points(raw: Any, fallback: int) -> int:
    try:
        return int(raw)
    except (TypeError, ValueError):
        return int(fallback)


def _coerce_strength(raw: Any, fallback: str) -> str:
    if raw is None:
        return fallback
    text = str(raw).strip().lower()
    if text in {"strong", "soft", "medium"}:
        return text
    return fallback


def _coerce_ai_needed(raw: Any, fallback: bool) -> bool:
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
    return fallback


@lru_cache(maxsize=1)
def load_validation_config(path: str | Path = _CONFIG_PATH) -> ValidationConfig:
    """Load validation metadata from YAML configuration."""

    config_path = Path(path)
    try:
        raw_text = config_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        logger.warning("VALIDATION_CONFIG_MISSING path=%s", config_path)
        defaults = ValidationRule("unknown", 3, tuple(), 3, "soft", False)
        return ValidationConfig(defaults=defaults, fields={})

    try:
        loaded = yaml.safe_load(raw_text) or {}
    except yaml.YAMLError:
        logger.exception("VALIDATION_CONFIG_PARSE_FAILED path=%s", config_path)
        loaded = {}

    defaults_raw = loaded.get("defaults") if isinstance(loaded, Mapping) else None
    if isinstance(defaults_raw, Mapping):
        default_category = _coerce_category(defaults_raw.get("category"), "unknown")
        default_min_days = _coerce_min_days(defaults_raw.get("min_days"), 3)
        default_documents = _coerce_documents(defaults_raw.get("documents"), ())
        default_points = _coerce_points(defaults_raw.get("points"), 3)
        default_strength = _coerce_strength(defaults_raw.get("strength"), "soft")
        default_ai_needed = _coerce_ai_needed(defaults_raw.get("ai_needed"), False)
    else:
        default_category = "unknown"
        default_min_days = 3
        default_documents = tuple()
        default_points = 3
        default_strength = "soft"
        default_ai_needed = False

    defaults = ValidationRule(
        default_category,
        default_min_days,
        default_documents,
        default_points,
        default_strength,
        default_ai_needed,
    )

    fields_cfg: Dict[str, ValidationRule] = {}
    fields_raw = loaded.get("fields") if isinstance(loaded, Mapping) else None
    if isinstance(fields_raw, Mapping):
        for key, value in fields_raw.items():
            if not isinstance(value, Mapping):
                continue
            category = _coerce_category(value.get("category"), defaults.category)
            min_days = _coerce_min_days(value.get("min_days"), defaults.min_days)
            documents = _coerce_documents(value.get("documents"), defaults.documents)
            points = _coerce_points(value.get("points"), defaults.points)
            strength = _coerce_strength(value.get("strength"), defaults.strength)
            ai_needed = _coerce_ai_needed(value.get("ai_needed"), defaults.ai_needed)
            fields_cfg[str(key)] = ValidationRule(
                category, min_days, documents, points, strength, ai_needed
            )

    return ValidationConfig(defaults=defaults, fields=fields_cfg)


def _filter_inconsistent_fields(
    field_consistency: Mapping[str, Any]
) -> Dict[str, Dict[str, Any]]:
    """Return only the inconsistent fields from a field consistency payload."""

    result: Dict[str, Dict[str, Any]] = {}
    for raw_field, raw_details in field_consistency.items():
        if not isinstance(raw_details, Mapping):
            continue

        consensus = str(raw_details.get("consensus", "")).lower()
        if consensus == "unanimous":
            continue

        field = str(raw_field)
        normalized = raw_details.get("normalized")
        if isinstance(normalized, Mapping):
            normalized_payload = dict(normalized)
        else:
            normalized_payload = normalized

        raw_values = raw_details.get("raw")
        if isinstance(raw_values, Mapping):
            raw_payload = dict(raw_values)
        else:
            raw_payload = raw_values

        disagreeing = raw_details.get("disagreeing_bureaus") or []
        if isinstance(disagreeing, Sequence) and not isinstance(
            disagreeing, (str, bytes, bytearray)
        ):
            disagreeing_list = sorted(str(item) for item in disagreeing)
        else:
            disagreeing_list = []

        missing = raw_details.get("missing_bureaus") or []
        if isinstance(missing, Sequence) and not isinstance(
            missing, (str, bytes, bytearray)
        ):
            missing_list = sorted(str(item) for item in missing)
        else:
            missing_list = []

        result[field] = {
            "normalized": normalized_payload,
            "raw": raw_payload,
            "consensus": raw_details.get("consensus"),
            "disagreeing_bureaus": disagreeing_list,
            "missing_bureaus": missing_list,
        }

    return result


_HISTORY_FIELDS = {"two_year_payment_history", "seven_year_history"}
_SEMANTIC_FIELDS = {"account_type", "creditor_type", "creditor_remarks"}


def _looks_like_date_field(field: str) -> bool:
    return "date" in field.lower()


def _is_numeric_value(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_numeric_field(field: str, normalized: Mapping[str, Any]) -> bool:
    if any(_is_numeric_value(value) for value in normalized.values() if value is not None):
        return True
    lowered = field.lower()
    if any(
        keyword in lowered
        for keyword in (
            "amount",
            "balance",
            "limit",
            "payment",
            "value",
            "due",
            "credit",
            "loan",
            "debt",
        )
    ):
        return True
    return False


def _has_missing_mismatch(normalized: Mapping[str, Any]) -> bool:
    values = list(normalized.values())
    return any(value is None for value in values) and any(
        value is not None for value in values
    )


def _determine_account_number_strength(normalized: Mapping[str, Any]) -> tuple[str, bool]:
    last4_values = set()
    for value in normalized.values():
        if isinstance(value, Mapping):
            last4 = value.get("last4")
        else:
            last4 = None
        if last4:
            last4_values.add(str(last4))
    if len(last4_values) > 1:
        return "strong", False
    return "soft", True


def _determine_dispute_strength(normalized: Mapping[str, Any]) -> tuple[str, bool]:
    seen = {str(value) for value in normalized.values() if value is not None}
    if len(seen) > 1 or _has_missing_mismatch(normalized):
        return "strong", False
    return "strong", False


def _apply_strength_policy(
    field: str, details: Mapping[str, Any], rule: ValidationRule
) -> ValidationRule:
    normalized = details.get("normalized")
    if not isinstance(normalized, Mapping):
        normalized = {}

    strength = rule.strength
    ai_needed = rule.ai_needed

    if field in _HISTORY_FIELDS:
        strength, ai_needed = "strong", False
    elif field == "account_number_display":
        strength, ai_needed = _determine_account_number_strength(normalized)
    elif field in _SEMANTIC_FIELDS:
        strength, ai_needed = "soft", True
    elif _looks_like_date_field(field):
        strength, ai_needed = "strong", False
    elif _is_numeric_field(field, normalized):
        strength, ai_needed = "strong", False
    elif field == "dispute_status":
        strength, ai_needed = _determine_dispute_strength(normalized)
    elif _has_missing_mismatch(normalized) and strength != "strong":
        strength = "medium"

    if strength == rule.strength and ai_needed == rule.ai_needed:
        return rule

    return ValidationRule(
        rule.category,
        rule.min_days,
        rule.documents,
        rule.points,
        strength,
        ai_needed,
    )


def _select_requirement_bureaus(
    details: Mapping[str, Any], *, broadcast_all: bool
) -> List[str]:
    disagreeing = details.get("disagreeing_bureaus")
    if not isinstance(disagreeing, Sequence) or isinstance(disagreeing, (str, bytes, bytearray)):
        disagreeing_list: List[str] = []
    else:
        disagreeing_list = [str(item) for item in disagreeing]

    normalized = details.get("normalized")
    if not isinstance(normalized, Mapping):
        normalized = {}

    if broadcast_all:
        bureaus = sorted(str(key) for key in normalized.keys())
        if bureaus:
            return bureaus
    if disagreeing_list:
        return sorted(set(disagreeing_list))

    missing = details.get("missing_bureaus")
    if isinstance(missing, Sequence) and not isinstance(missing, (str, bytes, bytearray)):
        missing_list = [str(item) for item in missing]
        if missing_list:
            return sorted(set(missing_list))
    return sorted(str(key) for key in normalized.keys())


def _build_requirement_entries(
    fields: Mapping[str, Any],
    config: ValidationConfig,
    *,
    broadcast_all: bool,
) -> List[Dict[str, Any]]:
    requirements: List[Dict[str, Any]] = []
    for field in sorted(fields.keys()):
        details = fields[field]
        rule = config.fields.get(field, config.defaults)
        rule = _apply_strength_policy(field, details, rule)
        bureaus = _select_requirement_bureaus(details, broadcast_all=broadcast_all)
        requirements.append(
            {
                "field": field,
                "category": rule.category,
                "min_days": rule.min_days,
                "documents": list(rule.documents),
                "strength": rule.strength,
                "ai_needed": rule.ai_needed,
                "bureaus": bureaus,
            }
        )
    return requirements


def build_validation_requirements(
    bureaus: Mapping[str, Mapping[str, Any]],
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Return validation requirements for fields with cross-bureau inconsistencies."""

    config = load_validation_config()
    field_consistency = compute_field_consistency(bureaus)
    inconsistencies = _filter_inconsistent_fields(field_consistency)
    broadcast_all = os.getenv("BROADCAST_DISPUTES", "1") == "1"
    requirements = _build_requirement_entries(
        inconsistencies, config, broadcast_all=broadcast_all
    )

    return requirements, inconsistencies


def build_summary_payload(
    requirements: Sequence[Mapping[str, Any]],
    *,
    field_consistency: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Build the summary.json payload for validation requirements."""

    entries = [dict(item) for item in requirements]
    payload = {"requirements": entries, "count": len(entries)}
    if field_consistency:
        payload["field_consistency"] = dict(field_consistency)
    return payload


def _load_summary(summary_path: Path) -> MutableMapping[str, Any]:
    try:
        raw = summary_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    except OSError:
        return {}
    try:
        loaded = json.loads(raw)
    except Exception:
        return {}
    if not isinstance(loaded, Mapping):
        return {}
    return dict(loaded)


def apply_validation_summary(
    summary_path: Path,
    payload: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Update ``summary.json`` with validation requirements when they changed."""

    summary_data = _load_summary(summary_path)
    existing = summary_data.get("validation_requirements")

    count = int(payload.get("count") or 0)
    if count <= 0:
        if "validation_requirements" in summary_data:
            summary_data.pop("validation_requirements", None)
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            if os.getenv("COMPACT_MERGE_SUMMARY", "1") == "1":
                compact_merge_sections(summary_data)
            _atomic_write_json(summary_path, summary_data)
        return summary_data

    if not isinstance(existing, Mapping) or dict(existing) != dict(payload):
        summary_data["validation_requirements"] = dict(payload)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        if os.getenv("COMPACT_MERGE_SUMMARY", "1") == "1":
            compact_merge_sections(summary_data)
        _atomic_write_json(summary_path, summary_data)

    return summary_data


def sync_validation_tag(
    tag_path: Path,
    fields: Sequence[str],
    *,
    emit: bool,
) -> None:
    """Ensure ``tags.json`` reflects the validation requirements state."""

    try:
        tags = read_tags(tag_path)
    except ValueError:
        logger.exception("VALIDATION_TAG_READ_FAILED path=%s", tag_path)
        return

    filtered = [tag for tag in tags if tag.get("kind") != _VALIDATION_TAG_KIND]
    changed = len(filtered) != len(tags)

    if emit and fields:
        timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        entry = {
            "kind": _VALIDATION_TAG_KIND,
            "fields": list(fields),
            "at": timestamp,
        }
        filtered.append(entry)
        changed = True

    if changed:
        try:
            write_tags_atomic(tag_path, filtered)
        except Exception:  # pragma: no cover - defensive file IO
            logger.exception("VALIDATION_TAG_WRITE_FAILED path=%s", tag_path)


def _should_emit_tags() -> bool:
    return os.environ.get("WRITE_VALIDATION_TAGS") == "1"


def build_validation_requirements_for_account(account_dir: str | Path) -> Dict[str, Any]:
    """Compute and persist validation requirements for ``account_dir``."""

    account_path = Path(account_dir)
    bureaus_path = account_path / "bureaus.json"
    summary_path = account_path / "summary.json"
    tags_path = account_path / "tags.json"

    if not bureaus_path.exists():
        logger.debug(
            "VALIDATION_REQUIREMENTS_SKIP_NO_BUREAUS path=%s", bureaus_path
        )
        return {"status": "no_bureaus_json"}

    try:
        raw_text = bureaus_path.read_text(encoding="utf-8")
    except OSError:
        logger.warning(
            "VALIDATION_REQUIREMENTS_READ_FAILED path=%s", bureaus_path, exc_info=True
        )
        return {"status": "invalid_bureaus_json"}

    try:
        bureaus_raw = json.loads(raw_text)
    except json.JSONDecodeError:
        logger.warning(
            "VALIDATION_REQUIREMENTS_INVALID_JSON path=%s", bureaus_path, exc_info=True
        )
        return {"status": "invalid_bureaus_json"}

    if not isinstance(bureaus_raw, Mapping):
        logger.warning(
            "VALIDATION_REQUIREMENTS_INVALID_TYPE path=%s type=%s",
            bureaus_path,
            type(bureaus_raw).__name__,
        )
        return {"status": "invalid_bureaus_json"}

    field_consistency_full = compute_field_consistency(bureaus_raw)

    config = load_validation_config()
    inconsistencies = _filter_inconsistent_fields(field_consistency_full)
    broadcast_all = os.getenv("BROADCAST_DISPUTES", "1") == "1"
    requirements = _build_requirement_entries(
        inconsistencies, config, broadcast_all=broadcast_all
    )
    payload = build_summary_payload(
        requirements, field_consistency=inconsistencies
    )
    summary_after = apply_validation_summary(summary_path, payload)

    debug_enabled = os.getenv("VALIDATION_DEBUG_DUMP") == "1"
    debug_key = "validation_debug"

    if debug_enabled:
        field_to_bureau: Dict[str, Any] = {}
        for field, details in field_consistency_full.items():
            if not isinstance(details, Mapping):
                continue
            raw_map = details.get("raw")
            if isinstance(raw_map, Mapping):
                entries = sorted(raw_map.items(), key=lambda item: str(item[0]))
                field_to_bureau[str(field)] = {
                    str(bureau): value for bureau, value in entries
                }
        debug_payload = {"raw_snapshot": {"field_to_bureau": field_to_bureau}}
        if summary_after.get(debug_key) != debug_payload:
            summary_after[debug_key] = debug_payload
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            if os.getenv("COMPACT_MERGE_SUMMARY", "1") == "1":
                compact_merge_sections(summary_after)
            _atomic_write_json(summary_path, summary_after)
    else:
        if debug_key in summary_after:
            summary_after.pop(debug_key, None)
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            if os.getenv("COMPACT_MERGE_SUMMARY", "1") == "1":
                compact_merge_sections(summary_after)
            _atomic_write_json(summary_path, summary_after)

    fields = [
        str(entry.get("field")) for entry in requirements if entry.get("field")
    ]
    sync_validation_tag(tags_path, fields, emit=_should_emit_tags())

    return {
        "status": "ok",
        "count": len(requirements),
        "fields": fields,
        "validation_requirements": payload,
    }

