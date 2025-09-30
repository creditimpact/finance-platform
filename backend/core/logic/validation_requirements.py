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
from backend.core.logic.consistency import compute_inconsistent_fields

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


@lru_cache(maxsize=1)
def load_validation_config(path: str | Path = _CONFIG_PATH) -> ValidationConfig:
    """Load validation metadata from YAML configuration."""

    config_path = Path(path)
    try:
        raw_text = config_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        logger.warning("VALIDATION_CONFIG_MISSING path=%s", config_path)
        defaults = ValidationRule("unspecified", 3, tuple())
        return ValidationConfig(defaults=defaults, fields={})

    try:
        loaded = yaml.safe_load(raw_text) or {}
    except yaml.YAMLError:
        logger.exception("VALIDATION_CONFIG_PARSE_FAILED path=%s", config_path)
        loaded = {}

    defaults_raw = loaded.get("defaults") if isinstance(loaded, Mapping) else None
    if isinstance(defaults_raw, Mapping):
        default_category = _coerce_category(defaults_raw.get("category"), "unspecified")
        default_min_days = _coerce_min_days(defaults_raw.get("min_days"), 3)
        default_documents = _coerce_documents(defaults_raw.get("documents"), ())
    else:
        default_category = "unspecified"
        default_min_days = 3
        default_documents = tuple()

    defaults = ValidationRule(default_category, default_min_days, default_documents)

    fields_cfg: Dict[str, ValidationRule] = {}
    fields_raw = loaded.get("fields") if isinstance(loaded, Mapping) else None
    if isinstance(fields_raw, Mapping):
        for key, value in fields_raw.items():
            if not isinstance(value, Mapping):
                continue
            category = _coerce_category(value.get("category"), defaults.category)
            min_days = _coerce_min_days(value.get("min_days"), defaults.min_days)
            documents = _coerce_documents(value.get("documents"), defaults.documents)
            fields_cfg[str(key)] = ValidationRule(category, min_days, documents)

    return ValidationConfig(defaults=defaults, fields=fields_cfg)


def build_validation_requirements(
    bureaus: Mapping[str, Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Return validation requirements for fields with cross-bureau inconsistencies."""

    config = load_validation_config()
    inconsistencies = compute_inconsistent_fields(bureaus)

    requirements: List[Dict[str, Any]] = []
    for field in sorted(inconsistencies.keys()):
        rule = config.fields.get(field, config.defaults)
        requirements.append(
            {
                "field": field,
                "category": rule.category,
                "min_days": rule.min_days,
                "documents": list(rule.documents),
            }
        )

    return requirements


def build_summary_payload(requirements: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Build the summary.json payload for validation requirements."""

    entries = [dict(item) for item in requirements]
    return {"requirements": entries, "count": len(entries)}


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
            _atomic_write_json(summary_path, summary_data)
        return summary_data

    if not isinstance(existing, Mapping) or dict(existing) != dict(payload):
        summary_data["validation_requirements"] = dict(payload)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
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

    requirements = build_validation_requirements(bureaus_raw)
    payload = build_summary_payload(requirements)
    apply_validation_summary(summary_path, payload)

    fields = [str(entry.get("field")) for entry in requirements if entry.get("field")]
    sync_validation_tag(tags_path, fields, emit=_should_emit_tags())

    return {"status": "ok", "count": len(requirements), "fields": fields}

