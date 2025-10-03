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

from backend.ai.validation_builder import build_validation_pack_for_account
from backend.core.io.json_io import _atomic_write_json
from backend.core.io.tags import read_tags, write_tags_atomic
from backend.core.logic.consistency import compute_field_consistency
from backend.core.logic.reason_classifier import classify_reason, decide_send_to_ai
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


def _is_validation_reason_enabled() -> bool:
    """Return ``True`` when reason enrichment should be applied."""

    raw_value = os.getenv("VALIDATION_REASON_ENABLED", "1")
    if raw_value is None:
        return True

    normalized = raw_value.strip().lower()
    return normalized in {"1", "true", "yes", "y", "on"}


@dataclass(frozen=True)
class ValidationRule:
    """Metadata describing the validation needed for a field."""

    category: str
    min_days: int
    documents: tuple[str, ...]
    points: int
    strength: str
    ai_needed: bool
    min_corroboration: int
    conditional_gate: bool


@dataclass(frozen=True)
class CategoryRule:
    """Fallback configuration scoped to a category."""

    min_days: int
    documents: tuple[str, ...]


@dataclass(frozen=True)
class ValidationConfig:
    defaults: ValidationRule
    fields: Mapping[str, ValidationRule]
    category_defaults: Mapping[str, CategoryRule]
    schema_version: int
    mode: str
    broadcast_disputes: bool
    threshold_points: int


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


def _coerce_min_corroboration(raw: Any, fallback: int) -> int:
    try:
        value = int(raw)
    except (TypeError, ValueError):
        value = int(fallback)
    return max(1, value)


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


def _coerce_bool(raw: Any, fallback: bool) -> bool:
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
        defaults = ValidationRule("unknown", 3, tuple(), 3, "soft", False, 1, False)
        return ValidationConfig(
            defaults=defaults,
            fields={},
            category_defaults={},
            schema_version=1,
            mode="broad",
            broadcast_disputes=True,
            threshold_points=45,
        )

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
        default_min_corroboration = _coerce_min_corroboration(
            defaults_raw.get("min_corroboration"), 1
        )
        default_conditional_gate = _coerce_bool(
            defaults_raw.get("conditional_gate"), False
        )
    else:
        default_category = "unknown"
        default_min_days = 3
        default_documents = tuple()
        default_points = 3
        default_strength = "soft"
        default_ai_needed = False
        default_min_corroboration = 1
        default_conditional_gate = False

    defaults = ValidationRule(
        default_category,
        default_min_days,
        default_documents,
        default_points,
        default_strength,
        default_ai_needed,
        default_min_corroboration,
        default_conditional_gate,
    )

    category_defaults_raw = (
        loaded.get("category_defaults") if isinstance(loaded, Mapping) else None
    )
    category_defaults: Dict[str, CategoryRule] = {}
    if isinstance(category_defaults_raw, Mapping):
        for key, value in category_defaults_raw.items():
            if not isinstance(value, Mapping):
                continue
            min_days = _coerce_min_days(value.get("min_days"), defaults.min_days)
            documents = _coerce_documents(value.get("documents"), defaults.documents)
            category_defaults[str(key)] = CategoryRule(min_days=min_days, documents=documents)

    def _resolve_field_rule(field: str, value: Mapping[str, Any]) -> ValidationRule:
        category = _coerce_category(value.get("category"), defaults.category)
        category_fallback = category_defaults.get(category)
        fallback_min_days = (
            category_fallback.min_days if category_fallback else defaults.min_days
        )
        fallback_documents = (
            category_fallback.documents
            if category_fallback and category_fallback.documents
            else defaults.documents
        )
        min_days = _coerce_min_days(value.get("min_days"), fallback_min_days)
        documents = _coerce_documents(value.get("documents"), fallback_documents)
        points = _coerce_points(value.get("points"), defaults.points)
        strength = _coerce_strength(value.get("strength"), defaults.strength)
        ai_needed = _coerce_ai_needed(value.get("ai_needed"), defaults.ai_needed)
        min_corroboration = _coerce_min_corroboration(
            value.get("min_corroboration"), defaults.min_corroboration
        )
        conditional_gate = _coerce_bool(
            value.get("conditional_gate"), defaults.conditional_gate
        )
        return ValidationRule(
            category,
            min_days,
            documents,
            points,
            strength,
            ai_needed,
            min_corroboration,
            conditional_gate,
        )

    fields_cfg: Dict[str, ValidationRule] = {}
    fields_raw = loaded.get("fields") if isinstance(loaded, Mapping) else None
    if isinstance(fields_raw, Mapping):
        for key, value in fields_raw.items():
            if not isinstance(value, Mapping):
                continue
            fields_cfg[str(key)] = _resolve_field_rule(str(key), value)

    schema_version = _coerce_points(loaded.get("schema_version"), 1)
    mode_raw = str(loaded.get("mode", "broad")) if isinstance(loaded, Mapping) else "broad"
    mode = mode_raw.strip().lower()
    if mode not in {"broad", "strict"}:
        mode = "broad"
    broadcast_default = True if mode == "broad" else False
    broadcast_disputes = _coerce_bool(
        loaded.get("broadcast_disputes"), broadcast_default
    )
    threshold_points = _coerce_points(loaded.get("threshold_points"), 45)

    return ValidationConfig(
        defaults=defaults,
        fields=fields_cfg,
        category_defaults=category_defaults,
        schema_version=schema_version,
        mode=mode,
        broadcast_disputes=broadcast_disputes,
        threshold_points=threshold_points,
    )


def _clone_field_consistency(
    field_consistency: Mapping[str, Any]
) -> Dict[str, Any]:
    """Deep-copy ``field_consistency`` ensuring plain ``dict``/``list`` containers."""

    def _clone(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {str(key): _clone(item) for key, item in value.items()}
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            return [_clone(item) for item in value]
        return value

    return {str(field): _clone(details) for field, details in field_consistency.items()}


def _strip_raw_from_field_consistency(
    field_consistency: Mapping[str, Any]
) -> Dict[str, Any]:
    """Return a deep copy of ``field_consistency`` without bureau raw values."""

    cloned = _clone_field_consistency(field_consistency)

    for field, details in list(cloned.items()):
        if isinstance(details, dict):
            details.pop("raw", None)

    return cloned


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

_HISTORY_REQUIREMENT_OVERRIDES: Mapping[str, ValidationRule] = {
    "two_year_payment_history": ValidationRule(
        category="history",
        min_days=18,
        documents=("monthly_statements_2y", "internal_payment_history"),
        points=10,
        strength="strong",
        ai_needed=False,
        min_corroboration=1,
        conditional_gate=False,
    ),
    "seven_year_history": ValidationRule(
        category="history",
        min_days=25,
        documents=(
            "cra_report_7y",
            "cra_audit_logs",
            "collection_history",
        ),
        points=12,
        strength="strong",
        ai_needed=False,
        min_corroboration=1,
        conditional_gate=False,
    ),
}

_DELINQUENCY_MARKERS = {
    "30",
    "60",
    "90",
    "120",
    "150",
    "180",
    "CO",
    "LATE30",
    "LATE60",
    "LATE90",
}


def _coerce_history_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _is_structured_history_value(field: str, value: Any) -> bool:
    if value is None:
        return False
    if field == "two_year_payment_history":
        if isinstance(value, Mapping):
            return True
        if isinstance(value, Sequence) and not isinstance(
            value, (str, bytes, bytearray)
        ):
            return True
        return False
    if field == "seven_year_history":
        return isinstance(value, Mapping)
    return True


def _history_counts_signature(field: str, value: Any) -> tuple[int, ...]:
    if not isinstance(value, Mapping):
        return ()
    if field == "two_year_payment_history":
        counts = value.get("counts")
        if not isinstance(counts, Mapping):
            return ()
        return (
            _coerce_history_int(counts.get("CO")),
            _coerce_history_int(counts.get("late30")),
            _coerce_history_int(counts.get("late60")),
            _coerce_history_int(counts.get("late90")),
        )
    if field == "seven_year_history":
        return (
            _coerce_history_int(value.get("late30")),
            _coerce_history_int(value.get("late60")),
            _coerce_history_int(value.get("late90")),
        )
    return ()


def _history_tokens_signature(field: str, value: Any) -> tuple[str, ...]:
    if field != "two_year_payment_history":
        return ()
    if not isinstance(value, Mapping):
        return ()
    tokens = value.get("tokens")
    if not isinstance(tokens, Sequence) or isinstance(tokens, (str, bytes, bytearray)):
        return ()
    signature: list[str] = []
    for token in tokens:
        if token is None:
            continue
        if isinstance(token, Mapping):
            status = token.get("status")
            if status is not None:
                text = str(status).strip().upper()
                if text:
                    signature.append(text)
                continue
            serialized = json.dumps(token, sort_keys=True)
            if serialized:
                signature.append(serialized.upper())
            continue
        text = str(token).strip()
        if text:
            signature.append(text.upper())
    return tuple(signature)


def _history_signature_has_delinquency(signature: Sequence[str]) -> bool:
    for token in signature:
        normalized = str(token).strip().upper()
        if not normalized:
            continue
        condensed = normalized.replace(" ", "")
        if condensed in _DELINQUENCY_MARKERS:
            return True
        if any(marker in normalized for marker in {"CHARGE", "COLLECT", "DEROG"}):
            return True
    return False


def _determine_history_strength(
    field: str, details: Mapping[str, Any]
) -> tuple[str, bool]:
    normalized = details.get("normalized")
    if not isinstance(normalized, Mapping):
        return "strong", False

    missing_raw = details.get("missing_bureaus") or []
    if isinstance(missing_raw, Sequence) and not isinstance(
        missing_raw, (str, bytes, bytearray)
    ):
        missing = {str(bureau) for bureau in missing_raw}
    else:
        missing = set()

    present_bureaus = [
        str(bureau)
        for bureau in normalized.keys()
        if str(bureau) not in missing
    ]
    if missing and present_bureaus:
        return "soft", True

    if not present_bureaus:
        return "strong", False

    raw_values = details.get("raw")
    raw_map = raw_values if isinstance(raw_values, Mapping) else {}

    for bureau in present_bureaus:
        if not _is_structured_history_value(field, raw_map.get(bureau)):
            return "soft", True

    counts_signatures = [
        _history_counts_signature(field, normalized.get(bureau))
        for bureau in present_bureaus
    ]
    if field == "two_year_payment_history":
        unique_counts = {signature for signature in counts_signatures}
        if len(unique_counts) <= 1:
            token_signatures = [
                _history_tokens_signature(field, normalized.get(bureau))
                for bureau in present_bureaus
            ]
            unique_tokens = {signature for signature in token_signatures}
            if len(unique_tokens) > 1:
                if not any(
                    _history_signature_has_delinquency(signature)
                    for signature in unique_tokens
                ):
                    return "soft", True

    return "strong", False
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
        strength, ai_needed = _determine_history_strength(field, details)
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
        rule.min_corroboration,
        rule.conditional_gate,
    )


def _resolve_validation_mode(config: ValidationConfig) -> str:
    override = os.getenv("VALIDATION_MODE")
    if override:
        lowered = override.strip().lower()
        if lowered in {"broad", "strict"}:
            return lowered
    return config.mode


def _should_broadcast(config: ValidationConfig) -> bool:
    override = os.getenv("BROADCAST_DISPUTES")
    if override is not None:
        return override.strip() == "1"

    mode = _resolve_validation_mode(config)
    if mode == "broad":
        return True

    return config.broadcast_disputes


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

    missing = details.get("missing_bureaus")
    if isinstance(missing, Sequence) and not isinstance(missing, (str, bytes, bytearray)):
        missing_list = [str(item) for item in missing]
    else:
        missing_list = []

    participants = set(disagreeing_list)
    participants.update(missing_list)

    if missing_list:
        present_bureaus = [
            str(bureau)
            for bureau, value in normalized.items()
            if value is not None and str(bureau) not in missing_list
        ]
        participants.update(present_bureaus)

    if not participants:
        participants.update(str(key) for key in normalized.keys())

    return sorted(participants)


def _build_requirement_entries(
    fields: Mapping[str, Any],
    config: ValidationConfig,
    *,
    broadcast_all: bool,
) -> List[Dict[str, Any]]:
    requirements: List[Dict[str, Any]] = []
    for field in sorted(fields.keys()):
        details = fields[field]
        base_rule = config.fields.get(field)
        if base_rule is None and field not in _HISTORY_REQUIREMENT_OVERRIDES:
            continue
        rule = base_rule or config.defaults

        if field in _HISTORY_REQUIREMENT_OVERRIDES:
            override = _HISTORY_REQUIREMENT_OVERRIDES[field]
            rule = ValidationRule(
                category=override.category,
                min_days=override.min_days,
                documents=override.documents,
                points=override.points,
                strength=override.strength,
                ai_needed=override.ai_needed,
                min_corroboration=override.min_corroboration,
                conditional_gate=override.conditional_gate,
            )

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
                "min_corroboration": rule.min_corroboration,
                "conditional_gate": rule.conditional_gate,
                "bureaus": bureaus,
            }
        )
    return requirements


def build_validation_requirements(
    bureaus: Mapping[str, Mapping[str, Any]],
    *,
    field_consistency: Mapping[str, Any] | None = None,
) -> tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
    """Return validation requirements for fields with cross-bureau inconsistencies."""

    config = load_validation_config()
    if not isinstance(field_consistency, Mapping):
        field_consistency_full = compute_field_consistency(dict(bureaus))
    else:
        field_consistency_full = _clone_field_consistency(field_consistency)

    inconsistencies = _filter_inconsistent_fields(field_consistency_full)
    broadcast_all = _should_broadcast(config)
    requirements = _build_requirement_entries(
        inconsistencies, config, broadcast_all=broadcast_all
    )

    return requirements, inconsistencies, field_consistency_full


def _coerce_normalized_map(details: Mapping[str, Any] | None) -> Dict[str, Any]:
    """Extract a mapping of bureau values from ``details`` safely."""

    if not isinstance(details, Mapping):
        return {}

    normalized = details.get("normalized")
    if isinstance(normalized, Mapping):
        return {str(key): value for key, value in normalized.items()}

    return {}


def _build_finding(
    entry: Mapping[str, Any],
    field_consistency: Mapping[str, Any] | None,
) -> Dict[str, Any]:
    """Return a finding enriched with reason metadata and AI routing."""

    finding = dict(entry)

    field_name = finding.get("field")
    details: Mapping[str, Any] | None = None
    if isinstance(field_consistency, Mapping) and isinstance(field_name, str):
        candidate = field_consistency.get(field_name)
        if isinstance(candidate, Mapping):
            details = candidate

    reason_details = classify_reason(_coerce_normalized_map(details))

    finding.update(reason_details)
    finding["send_to_ai"] = decide_send_to_ai(field_name, reason_details)

    return finding


def build_summary_payload(
    requirements: Sequence[Mapping[str, Any]],
    *,
    field_consistency: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Build the summary.json payload for validation requirements."""

    normalized_requirements = [dict(entry) for entry in requirements]
    reasons_enabled = _is_validation_reason_enabled()

    include_requirements = os.getenv(
        "VALIDATION_SUMMARY_INCLUDE_REQUIREMENTS", "0"
    )
    include_requirements_flag = str(include_requirements).strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
        "on",
    }

    if reasons_enabled:
        findings = [
            _build_finding(entry, field_consistency)
            for entry in normalized_requirements
        ]
    else:
        findings = [dict(entry) for entry in normalized_requirements]

    payload: Dict[str, Any] = {
        "findings": findings,
        "count": len(findings),
    }

    if include_requirements_flag:
        payload["requirements"] = normalized_requirements

    if field_consistency:
        if reasons_enabled:
            sanitized_consistency = _strip_raw_from_field_consistency(field_consistency)
            if sanitized_consistency:
                payload["field_consistency"] = sanitized_consistency
        else:
            cloned_consistency = _clone_field_consistency(field_consistency)
            if cloned_consistency:
                payload["field_consistency"] = cloned_consistency

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
    field_consistency = payload.get("field_consistency")
    has_consistency = isinstance(field_consistency, Mapping) and bool(field_consistency)
    if count <= 0 and not has_consistency:
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

    summary_snapshot = _load_summary(summary_path)
    summary_consistency = summary_snapshot.get("field_consistency")
    consistency_override = (
        _clone_field_consistency(summary_consistency)
        if isinstance(summary_consistency, Mapping)
        else None
    )

    requirements, _, field_consistency_full = build_validation_requirements(
        bureaus_raw, field_consistency=consistency_override
    )
    payload = build_summary_payload(
        requirements, field_consistency=field_consistency_full
    )
    summary_after = apply_validation_summary(summary_path, payload)

    debug_enabled = os.getenv("VALIDATION_DEBUG") == "1"
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

    sid: str | None = None
    account_id: str | None = None
    try:
        sid = summary_path.parents[3].name
        account_id = summary_path.parent.name
    except IndexError:
        sid = None
        account_id = None

    if sid and account_id:
        try:
            build_validation_pack_for_account(
                sid,
                account_id,
                summary_path,
                bureaus_path,
            )
        except Exception:  # pragma: no cover - defensive logging
            logger.exception(
                "VALIDATION_PACK_BUILD_FAILED sid=%s account=%s summary=%s",
                sid,
                account_id,
                summary_path,
            )

    return {
        "status": "ok",
        "count": len(requirements),
        "fields": fields,
        "validation_requirements": payload,
    }

