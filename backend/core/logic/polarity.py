"""Deterministic polarity classifier for bureau field values."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
import ast
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Literal

import yaml


Polarity = Literal["good", "bad", "neutral", "unknown"]
Severity = Literal["low", "medium", "high"]

logger = logging.getLogger(__name__)

_ALLOWED_POLARITIES: set[str] = {"good", "bad", "neutral", "unknown"}
_ALLOWED_SEVERITIES: set[str] = {"low", "medium", "high"}

_POLARITY_CONFIG_PATH: Path = Path(__file__).with_name("polarity_config.yml")
_CONFIG_CACHE: tuple[float | None, Dict[str, Any]] | None = None


def load_polarity_config() -> Dict[str, Any]:
    """Load and cache the polarity configuration YAML file."""

    global _CONFIG_CACHE

    path = _POLARITY_CONFIG_PATH

    try:
        stat_result = path.stat()
    except FileNotFoundError:
        logger.warning("POLARITY_CONFIG_NOT_FOUND path=%s", path)
        _CONFIG_CACHE = (None, {})
        return {}

    mtime = stat_result.st_mtime
    if _CONFIG_CACHE is not None and _CONFIG_CACHE[0] == mtime:
        return _CONFIG_CACHE[1]

    try:
        raw_text = path.read_text(encoding="utf-8")
    except OSError:
        logger.exception("POLARITY_CONFIG_READ_FAILED path=%s", path)
        _CONFIG_CACHE = (mtime, {})
        return {}

    try:
        data = yaml.safe_load(raw_text) or {}
    except yaml.YAMLError:
        logger.exception("POLARITY_CONFIG_PARSE_FAILED path=%s", path)
        data = {}

    if not isinstance(data, dict):
        data = {}

    _CONFIG_CACHE = (mtime, data)
    return data


def parse_money(raw: Any) -> Optional[float]:
    """Parse a currency value into a float (cents-aware)."""

    if raw is None:
        return None

    if isinstance(raw, (int, float)):
        return float(raw)

    text = str(raw).strip()
    if not text:
        return None

    text = text.replace("$", "").replace(",", "")
    text = text.replace(" ", "")
    if text.startswith("(") and text.endswith(")"):
        text = f"-{text[1:-1]}"

    text = text.replace("--", "")
    if not text or text in {"-", "-0", "0-"}:
        return None

    try:
        decimal_value = Decimal(text)
    except InvalidOperation:
        logger.debug("POLARITY_PARSE_MONEY_FAILED value=%r normalized=%r", raw, text)
        return None

    return float(decimal_value)


def is_blank(value: Any) -> bool:
    """Return True when ``value`` should be treated as blank."""

    if value is None:
        return True

    if isinstance(value, (int, float)):
        return False

    text = str(value).strip()
    return text == "" or text == "--"


def norm_text(value: Any) -> str:
    """Normalize ``value`` for keyword matching (casefold + collapse spaces)."""

    if value is None:
        return ""

    if isinstance(value, str):
        text = value
    else:
        text = str(value)

    collapsed = " ".join(text.split())
    return collapsed.casefold()


def _normalize_polarity(candidate: Any) -> Polarity:
    if isinstance(candidate, str):
        lowered = candidate.strip().lower()
        if lowered in _ALLOWED_POLARITIES:
            return lowered  # type: ignore[return-value]
    return "unknown"


def _normalize_severity(candidate: Any) -> Severity:
    if isinstance(candidate, str):
        lowered = candidate.strip().lower()
        if lowered in _ALLOWED_SEVERITIES:
            return lowered  # type: ignore[return-value]
    return "low"


def _safe_eval_boolean(expression: str, variables: Mapping[str, Any]) -> bool:
    """Evaluate ``expression`` using a whitelist of AST nodes."""

    try:
        parsed = ast.parse(expression, mode="eval")
    except SyntaxError:
        logger.debug("POLARITY_RULE_SYNTAX_ERROR expression=%r", expression)
        return False

    def _eval(node: ast.AST) -> Any:
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.BoolOp):
            values = [_eval(v) for v in node.values]
            if isinstance(node.op, ast.And):
                return all(values)
            if isinstance(node.op, ast.Or):
                return any(values)
            raise ValueError("unsupported bool op")
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            return not bool(_eval(node.operand))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            operand = _eval(node.operand)
            return +operand if isinstance(node.op, ast.UAdd) else -operand
        if isinstance(node, ast.Compare):
            left = _eval(node.left)
            for operator, comparator in zip(node.ops, node.comparators):
                right = _eval(comparator)
                if isinstance(operator, ast.Eq) and not (left == right):
                    return False
                elif isinstance(operator, ast.NotEq) and not (left != right):
                    return False
                elif isinstance(operator, ast.Gt) and not (left > right):
                    return False
                elif isinstance(operator, ast.GtE) and not (left >= right):
                    return False
                elif isinstance(operator, ast.Lt) and not (left < right):
                    return False
                elif isinstance(operator, ast.LtE) and not (left <= right):
                    return False
                left = right
            return True
        if isinstance(node, ast.Name):
            identifier = node.id.lower()
            if identifier == "true":
                return True
            if identifier == "false":
                return False
            if identifier in variables:
                return variables[identifier]
            raise ValueError(f"unknown identifier {node.id!r}")
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Num):  # pragma: no cover (legacy py compat)
            return node.n
        if isinstance(node, ast.Str):  # pragma: no cover
            return node.s
        raise ValueError(f"unsupported expression: {expression!r}")

    try:
        result = _eval(parsed)
    except Exception:
        logger.debug("POLARITY_RULE_EVAL_ERROR expression=%r", expression, exc_info=True)
        return False

    return bool(result)


def _evaluate_money(field_cfg: Mapping[str, Any], raw_value: Any) -> Dict[str, Any]:
    parsed_value = parse_money(raw_value)
    evidence: Dict[str, Any] = {"parsed": parsed_value}

    if parsed_value is None:
        return {
            "polarity": "unknown",
            "severity": "low",
            "evidence": evidence,
        }

    rules = field_cfg.get("rules")
    if isinstance(rules, Iterable):
        for rule in rules:
            if not isinstance(rule, Mapping):
                continue
            condition = rule.get("if")
            if not isinstance(condition, str) or not condition.strip():
                continue
            if _safe_eval_boolean(condition, {"value": parsed_value}):
                evidence["matched_rule"] = condition
                polarity = _normalize_polarity(rule.get("polarity"))
                severity = _normalize_severity(rule.get("severity"))
                return {
                    "polarity": polarity,
                    "severity": severity,
                    "evidence": evidence,
                }

    return {
        "polarity": "unknown",
        "severity": "low",
        "evidence": evidence,
    }


def _evaluate_date(field_cfg: Mapping[str, Any], raw_value: Any) -> Dict[str, Any]:
    present = not is_blank(raw_value)
    evidence: Dict[str, Any] = {"parsed": present}

    rules = field_cfg.get("rules")
    if isinstance(rules, Iterable):
        for rule in rules:
            if not isinstance(rule, Mapping):
                continue
            condition = rule.get("if")
            if not isinstance(condition, str) or not condition.strip():
                continue
            if _safe_eval_boolean(condition, {"is_present": present}):
                evidence["matched_rule"] = condition
                polarity = _normalize_polarity(rule.get("polarity"))
                severity = _normalize_severity(rule.get("severity"))
                return {
                    "polarity": polarity,
                    "severity": severity,
                    "evidence": evidence,
                }

    polarity = "neutral" if present else "unknown"
    return {
        "polarity": polarity,
        "severity": "low",
        "evidence": evidence,
    }


def _match_keyword(normalized_value: str, keywords: Iterable[Any]) -> Optional[str]:
    for keyword in keywords:
        if not isinstance(keyword, str):
            continue
        normalized_keyword = norm_text(keyword)
        if normalized_keyword and normalized_keyword in normalized_value:
            return keyword
    return None


def _evaluate_text(field_cfg: Mapping[str, Any], raw_value: Any) -> Dict[str, Any]:
    normalized_value = norm_text(raw_value)
    evidence: Dict[str, Any] = {"parsed": normalized_value or None}

    weights = field_cfg.get("weights")
    weight_map = weights if isinstance(weights, Mapping) else {}

    for category in ("bad", "good", "neutral"):
        keywords = field_cfg.get(f"{category}_keywords")
        if isinstance(keywords, Iterable):
            matched = _match_keyword(normalized_value, keywords)
            if matched:
                evidence["matched_keyword"] = matched
                polarity = _normalize_polarity(category)
                severity = _normalize_severity(weight_map.get(category))
                return {
                    "polarity": polarity,
                    "severity": severity,
                    "evidence": evidence,
                }

    default_polarity = _normalize_polarity(field_cfg.get("default"))
    severity = _normalize_severity(weight_map.get(default_polarity))
    return {
        "polarity": default_polarity,
        "severity": severity,
        "evidence": evidence,
    }


def classify_field_value(field: str, raw_value: Optional[str | int | float]) -> Dict[str, Any]:
    """Classify a field value using the polarity configuration."""

    config = load_polarity_config()
    fields_cfg = config.get("fields")
    if not isinstance(fields_cfg, Mapping):
        fields_cfg = {}

    field_cfg = fields_cfg.get(field)
    if not isinstance(field_cfg, Mapping):
        return {
            "polarity": "unknown",
            "severity": "low",
            "evidence": {"parsed": None},
        }

    field_type = str(field_cfg.get("type") or "").strip().lower()
    if field_type == "money":
        return _evaluate_money(field_cfg, raw_value)
    if field_type == "date":
        return _evaluate_date(field_cfg, raw_value)
    if field_type == "text":
        return _evaluate_text(field_cfg, raw_value)

    logger.debug("POLARITY_UNKNOWN_FIELD_TYPE field=%s type=%r", field, field_type)
    return {
        "polarity": "unknown",
        "severity": "low",
        "evidence": {"parsed": None},
    }


__all__ = [
    "Polarity",
    "Severity",
    "classify_field_value",
    "is_blank",
    "load_polarity_config",
    "norm_text",
    "parse_money",
]

