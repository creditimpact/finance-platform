"""Per-bureau polarity classification for tradeline fields."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml

import backend.config as config
from backend.core.io.tags import upsert_tag

logger = logging.getLogger(__name__)

_CONFIG_CACHE: tuple[dict[str, Any], str] | None = None
_ALLOWED_POLARITIES = {"good", "bad", "neutral", "unknown"}
_ALLOWED_SEVERITIES = {"low", "medium", "high"}
_POLARITY_TAG_KIND = "polarity_probe"
_POLARITY_TAG_UNIQUE_KEYS = ("kind", "bureau", "field")
_POLARITY_CONFIG_FILENAME = "polarity_config.yml"


@dataclass(frozen=True)
class PolarityResult:
    """Container for polarity check results."""

    processed_accounts: int
    updated_accounts: list[int]
    config_digest: str
    results: dict[int, dict[str, Any]]


def _load_config() -> tuple[dict[str, Any], str]:
    """Load polarity configuration from YAML, caching the result."""

    global _CONFIG_CACHE
    if _CONFIG_CACHE is not None:
        return _CONFIG_CACHE

    path = Path(__file__).with_name(_POLARITY_CONFIG_FILENAME)
    try:
        raw_text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        logger.warning("POLARITY_CONFIG_MISSING path=%s", path)
        _CONFIG_CACHE = ({}, "")
        return _CONFIG_CACHE

    try:
        data = yaml.safe_load(raw_text) or {}
    except yaml.YAMLError:
        logger.error("POLARITY_CONFIG_INVALID path=%s", path, exc_info=True)
        data = {}

    digest = sha256(raw_text.encode("utf-8")).hexdigest()[:16]
    _CONFIG_CACHE = (data, digest)
    return _CONFIG_CACHE


def _normalize_polarity(value: str | None) -> str:
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in _ALLOWED_POLARITIES:
            return lowered
    return "unknown"


def _normalize_severity(value: str | None) -> str:
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in _ALLOWED_SEVERITIES:
            return lowered
    return "low"


def _coerce_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _money_to_decimal(value: Any) -> Decimal | None:
    text = _coerce_text(value)
    if not text:
        return None

    normalized = text
    if normalized.startswith("--") and normalized.endswith("--"):
        normalized = normalized.strip("-")
    normalized = normalized.replace("$", "").replace(",", "")
    normalized = normalized.replace(" ", "")
    if not normalized:
        return None
    if normalized.startswith("(") and normalized.endswith(")"):
        normalized = f"-{normalized[1:-1]}"
    normalized = normalized.replace("--", "")
    if not normalized or normalized in {"-", "-0", "0-"}:
        return None

    try:
        return Decimal(normalized)
    except InvalidOperation:
        logger.debug("POLARITY_MONEY_PARSE_FAILED value=%r normalized=%r", value, normalized)
        return None


def _decimal_to_json(value: Decimal) -> int | float:
    try:
        integral = value.to_integral_value()
    except InvalidOperation:
        return float(value)
    if value == integral:
        return int(integral)
    return float(value)


def _evaluate_money(field_cfg: Mapping[str, Any], raw_value: Any) -> dict[str, Any]:
    numeric = _money_to_decimal(raw_value)
    result: dict[str, Any] = {"value": raw_value if raw_value is not None else None}
    if numeric is not None:
        result["numeric"] = _decimal_to_json(numeric)
    else:
        result["polarity"] = "unknown"
        result["severity"] = "low"
        result["reason"] = "value_missing_or_invalid"
        return result

    rules = field_cfg.get("rules")
    if isinstance(rules, Iterable):
        for entry in rules:
            if not isinstance(entry, Mapping):
                continue
            condition = _coerce_text(entry.get("if"))
            if not condition:
                continue
            try:
                outcome = bool(eval(condition, {"__builtins__": {}}, {"value": numeric}))
            except Exception:
                logger.debug(
                    "POLARITY_RULE_EVAL_FAILED field=%s rule=%r value=%s",
                    field_cfg,
                    condition,
                    numeric,
                    exc_info=True,
                )
                continue
            if not outcome:
                continue
            polarity = _normalize_polarity(entry.get("polarity"))
            severity = _normalize_severity(entry.get("severity"))
            result.update({"polarity": polarity, "severity": severity, "rule": condition})
            return result

    result.update({"polarity": "unknown", "severity": "low", "reason": "no_rule_matched"})
    return result


def _is_present(value: str) -> bool:
    if not value:
        return False
    lowered = value.lower()
    if lowered in {"--", "unknown", "n/a", "na"}:
        return False
    return True


def _evaluate_date(field_cfg: Mapping[str, Any], raw_value: Any) -> dict[str, Any]:
    text = _coerce_text(raw_value)
    present = _is_present(text)
    result: dict[str, Any] = {
        "value": text or None,
        "is_present": present,
    }

    rules = field_cfg.get("rules")
    if isinstance(rules, Iterable):
        for entry in rules:
            if not isinstance(entry, Mapping):
                continue
            condition = _coerce_text(entry.get("if"))
            if not condition:
                continue
            try:
                outcome = bool(eval(condition, {"__builtins__": {}}, {"is_present": present}))
            except Exception:
                logger.debug(
                    "POLARITY_RULE_EVAL_FAILED field=date rule=%r value=%s",
                    condition,
                    present,
                    exc_info=True,
                )
                continue
            if not outcome:
                continue
            polarity = _normalize_polarity(entry.get("polarity"))
            severity = _normalize_severity(entry.get("severity"))
            result.update({"polarity": polarity, "severity": severity, "rule": condition})
            return result

    result.update({"polarity": "unknown", "severity": "low", "reason": "no_rule_matched"})
    return result


def _match_keyword(value_lower: str, keywords: Iterable[str]) -> str | None:
    for keyword in keywords:
        if not isinstance(keyword, str):
            continue
        term = keyword.strip().lower()
        if term and term in value_lower:
            return keyword
    return None


def _evaluate_text(field_cfg: Mapping[str, Any], raw_value: Any) -> dict[str, Any]:
    text = _coerce_text(raw_value)
    lowered = text.lower()
    result: dict[str, Any] = {"value": text or None}

    for category in ("bad", "good", "neutral"):
        keywords = field_cfg.get(f"{category}_keywords") or []
        if not isinstance(keywords, Iterable):
            continue
        matched = _match_keyword(lowered, keywords)
        if matched:
            polarity = _normalize_polarity(category)
            weights = field_cfg.get("weights") or {}
            severity = _normalize_severity(weights.get(category))
            result.update({
                "polarity": polarity,
                "severity": severity,
                "matched_keyword": matched,
            })
            return result

    default_polarity = _normalize_polarity(field_cfg.get("default"))
    weights = field_cfg.get("weights") or {}
    severity = _normalize_severity(weights.get(default_polarity))
    result.update({"polarity": default_polarity, "severity": severity})
    return result


def _evaluate_field(field_cfg: Mapping[str, Any], raw_value: Any) -> dict[str, Any]:
    field_type = _coerce_text(field_cfg.get("type")).lower()
    if field_type == "money":
        return _evaluate_money(field_cfg, raw_value)
    if field_type == "date":
        return _evaluate_date(field_cfg, raw_value)
    if field_type == "text":
        return _evaluate_text(field_cfg, raw_value)

    logger.debug("POLARITY_UNKNOWN_FIELD_TYPE type=%r", field_type)
    return {"value": raw_value if raw_value is not None else None, "polarity": "unknown", "severity": "low"}


def _iter_bureaus(data: Mapping[str, Any], field_names: Sequence[str]) -> Iterable[tuple[str, Mapping[str, Any]]]:
    order = data.get("order")
    names: list[str] = []
    if isinstance(order, Sequence):
        for name in order:
            if isinstance(name, str):
                names.append(name)
    if not names:
        for name, payload in data.items():
            if not isinstance(name, str):
                continue
            if not isinstance(payload, Mapping):
                continue
            if any(field in payload for field in field_names):
                names.append(name)
    seen: set[str] = set()
    for name in names:
        if name in seen:
            continue
        seen.add(name)
        payload = data.get(name)
        if isinstance(payload, Mapping):
            yield name, payload


def _build_account_payload(
    bureaus_data: Mapping[str, Any],
    fields_cfg: Mapping[str, Any],
) -> dict[str, dict[str, dict[str, Any]]]:
    result: dict[str, dict[str, dict[str, Any]]] = {}
    field_names = list(fields_cfg.keys())
    for bureau_name, bureau_payload in _iter_bureaus(bureaus_data, field_names):
        bureau_result: dict[str, dict[str, Any]] = {}
        for field_name, field_cfg in fields_cfg.items():
            if not isinstance(field_cfg, Mapping):
                continue
            raw_value = bureau_payload.get(field_name)
            bureau_result[field_name] = _evaluate_field(field_cfg, raw_value)
        if bureau_result:
            result[bureau_name] = bureau_result
    return result


def _read_json(path: Path) -> Mapping[str, Any] | None:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        logger.debug("POLARITY_SOURCE_MISSING path=%s", path)
        return None
    except OSError:
        logger.warning("POLARITY_SOURCE_READ_FAILED path=%s", path, exc_info=True)
        return None

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        logger.warning("POLARITY_SOURCE_INVALID_JSON path=%s", path)
        return None

    if not isinstance(data, Mapping):
        logger.warning("POLARITY_SOURCE_INVALID_STRUCTURE path=%s", path)
        return None
    return data


def _write_summary(path: Path, payload: Mapping[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError:
        logger.warning("POLARITY_SUMMARY_WRITE_FAILED path=%s", path, exc_info=True)
        raise


def _maybe_apply_probe_tags(
    account_dir: Path,
    bureau_payload: Mapping[str, Any],
    *,
    config_digest: str,
    enable_probe: bool,
) -> None:
    if not enable_probe:
        return

    for bureau_name, fields in bureau_payload.items():
        if not isinstance(fields, Mapping):
            continue
        for field_name, result in fields.items():
            if not isinstance(result, Mapping):
                continue
            tag_payload: dict[str, Any] = {
                "kind": _POLARITY_TAG_KIND,
                "bureau": bureau_name,
                "field": field_name,
                "polarity": result.get("polarity", "unknown"),
                "severity": result.get("severity", "low"),
                "config_digest": config_digest,
            }
            if "value" in result:
                tag_payload["value"] = result.get("value")
            if "matched_keyword" in result:
                tag_payload["matched_keyword"] = result.get("matched_keyword")
            if "rule" in result:
                tag_payload["rule"] = result.get("rule")
            upsert_tag(account_dir, tag_payload, unique_keys=_POLARITY_TAG_UNIQUE_KEYS)


def apply_polarity_checks(
    accounts_dir: Path,
    indices: Sequence[int],
    *,
    enable_tags_probe: bool | None = None,
) -> PolarityResult:
    """Apply polarity checks for ``indices`` under ``accounts_dir``."""

    config_data, digest = _load_config()
    fields_cfg = config_data.get("fields")
    if not isinstance(fields_cfg, Mapping) or not fields_cfg:
        logger.info("POLARITY_NO_FIELDS_DEFINED")
        return PolarityResult(0, [], digest, {})

    if enable_tags_probe is None:
        enable_tags_probe = config.env_bool("ENABLE_POLARITY_TAG_PROBE", False)

    processed = 0
    updated: list[int] = []
    results: dict[int, dict[str, Any]] = {}

    for index in indices:
        try:
            account_dir = accounts_dir / str(index)
        except Exception:
            logger.debug("POLARITY_ACCOUNT_PATH_FAILED index=%r", index, exc_info=True)
            continue

        bureaus_path = account_dir / "bureaus.json"
        summary_path = account_dir / "summary.json"

        bureaus_data = _read_json(bureaus_path)
        if not bureaus_data:
            continue

        summary_data = _read_json(summary_path)
        if summary_data is None:
            summary_data = {}
        else:
            summary_data = dict(summary_data)

        bureau_payload = _build_account_payload(bureaus_data, fields_cfg)
        if not bureau_payload:
            continue

        processed += 1

        polarity_payload = {
            "schema_version": 1,
            "config_digest": digest,
            "bureaus": bureau_payload,
        }

        existing = summary_data.get("polarity_check")
        if existing != polarity_payload:
            summary_data["polarity_check"] = polarity_payload
            _write_summary(summary_path, summary_data)
            updated.append(int(index))

        results[int(index)] = polarity_payload

        try:
            _maybe_apply_probe_tags(
                account_dir,
                bureau_payload,
                config_digest=digest,
                enable_probe=bool(enable_tags_probe),
            )
        except Exception:
            logger.warning(
                "POLARITY_TAG_PROBE_FAILED account=%s", account_dir, exc_info=True
            )

    return PolarityResult(processed, updated, digest, results)
