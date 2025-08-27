import json
import os
from pathlib import Path
from typing import Any, Tuple


def env_bool(name: str, default: bool = False) -> bool:
    """Parse a boolean environment variable."""
    val = os.getenv(name)
    if val is None:
        return default
    return val.lower() in {"1", "true", "yes", "on"}


def env_str(name: str, default: str) -> str:
    """Fetch a string environment variable."""
    return os.getenv(name, default)


def env_float(name: str, default: float) -> float:
    """Parse a float environment variable."""
    val = os.getenv(name)
    if val is None:
        return default
    try:
        return float(val)
    except ValueError:
        return default


def env_int(name: str, default: int) -> int:
    """Parse an int environment variable."""
    val = os.getenv(name)
    if val is None:
        return default


def env_list(name: str, default: list[str]) -> list[str]:
    """Parse a comma-separated list environment variable."""
    val = os.getenv(name)
    if val is None:
        return default
    parts = [p.strip() for p in val.split(",") if p.strip()]
    return parts or default
    try:
        return int(val)
    except ValueError:
        return default


# Backwards compatibility for older imports
_env_bool = env_bool


def _load_keyword_lists() -> Tuple[dict, dict, dict]:
    """Load tiered keyword dictionaries from JSON/YAML config.

    Returns empty dictionaries when the config file is missing or malformed.
    """

    path = os.getenv("KEYWORDS_CONFIG_PATH", Path(__file__).with_name("keywords.json"))
    try:
        with open(path, "r", encoding="utf-8") as f:
            data: Any = json.load(f)
    except Exception:
        data = {}
    t1 = data.get("tier1") or data.get("TIER1_KEYWORDS") or {}
    t2 = data.get("tier2") or data.get("TIER2_KEYWORDS") or {}
    t3 = data.get("tier3") or data.get("TIER3_KEYWORDS") or {}
    return t1, t2, t3


UTILIZATION_PROBLEM_THRESHOLD = float(
    os.getenv("UTILIZATION_PROBLEM_THRESHOLD", "0.90")
)
SERIOUS_DELINQUENCY_MIN_DPD = int(os.getenv("SERIOUS_DELINQUENCY_MIN_DPD", "60"))

ENABLE_TIER1_KEYWORDS = _env_bool("ENABLE_TIER1_KEYWORDS", False)
ENABLE_TIER2_KEYWORDS = _env_bool("ENABLE_TIER2_KEYWORDS", False)
ENABLE_TIER3_KEYWORDS = _env_bool("ENABLE_TIER3_KEYWORDS", False)
ENABLE_TIER2_NUMERIC = _env_bool("ENABLE_TIER2_NUMERIC", True)

ENABLE_AI_ADJUDICATOR = env_bool("ENABLE_AI_ADJUDICATOR", False)
AI_MIN_CONFIDENCE = env_float("AI_MIN_CONFIDENCE", 0.70)
AI_MODEL_ID = env_str("AI_MODEL_ID", "gpt-4o-mini")
AI_TEMPERATURE_DEFAULT = env_float("AI_TEMPERATURE_DEFAULT", 0.0)
AI_REQUEST_TIMEOUT_S = env_int("AI_REQUEST_TIMEOUT_S", 8)
AI_MAX_TOKENS = env_int("AI_MAX_TOKENS", 600)
AI_MAX_RETRIES = env_int("AI_MAX_RETRIES", 1)
AI_HIERARCHY_VERSION = env_str("AI_HIERARCHY_VERSION", "v1")
AI_REDACT_STRATEGY = os.getenv("AI_REDACT_STRATEGY", "hash_last4")

_raw_t1, _raw_t2, _raw_t3 = _load_keyword_lists()

TIER1_KEYWORDS = _raw_t1 if ENABLE_TIER1_KEYWORDS else {}
TIER2_KEYWORDS = _raw_t2 if ENABLE_TIER2_KEYWORDS else {}
TIER3_KEYWORDS = _raw_t3 if ENABLE_TIER3_KEYWORDS else {}


# Case Store configuration
CASESTORE_DIR = env_str("CASESTORE_DIR", "/var/app/run/cases")
CASESTORE_REDACT_BEFORE_STORE = env_bool("CASESTORE_REDACT_BEFORE_STORE", True)
CASESTORE_ATOMIC_WRITES = env_bool("CASESTORE_ATOMIC_WRITES", True)
CASESTORE_VALIDATE_ON_LOAD = env_bool("CASESTORE_VALIDATE_ON_LOAD", True)

# Parser dual-write flags
ENABLE_CASESTORE_WRITE = env_bool("ENABLE_CASESTORE_WRITE", False)
CASESTORE_PARSER_LOG_PARITY = env_bool("CASESTORE_PARSER_LOG_PARITY", True)

# Stage A Case Store migration flags
ENABLE_CASESTORE_STAGEA = env_bool("ENABLE_CASESTORE_STAGEA", True)

# Stage A detection mode
PROBLEM_DETECTION_ONLY = env_bool("PROBLEM_DETECTION_ONLY", True)

# Candidate token logging
ENABLE_CANDIDATE_TOKEN_LOGGER = env_bool("ENABLE_CANDIDATE_TOKEN_LOGGER", True)
CANDIDATE_LOG_FORMAT = env_str("CANDIDATE_LOG_FORMAT", "jsonl")

# API decision metadata exposure
API_INCLUDE_DECISION_META = env_bool("API_INCLUDE_DECISION_META", True)
API_DECISION_META_MAX_FIELDS_USED = env_int("API_DECISION_META_MAX_FIELDS_USED", 6)

# Cross-bureau resolution
ENABLE_CROSS_BUREAU_RESOLUTION = env_bool("ENABLE_CROSS_BUREAU_RESOLUTION", False)
API_AGGREGATION_ID_STRATEGY = env_str("API_AGGREGATION_ID_STRATEGY", "winner")
API_INCLUDE_AGG_MEMBERS_META = env_bool("API_INCLUDE_AGG_MEMBERS_META", False)

# Parser audit instrumentation
PARSER_AUDIT_ENABLED = env_bool("PARSER_AUDIT_ENABLED", True)
PDF_TEXT_MIN_CHARS_PER_PAGE = env_int("PDF_TEXT_MIN_CHARS_PER_PAGE", 64)

# OCR fallback configuration
OCR_ENABLED = env_bool("OCR_ENABLED", False)
OCR_PROVIDER = env_str("OCR_PROVIDER", "tesseract")
OCR_TIMEOUT_MS = env_int("OCR_TIMEOUT_MS", 8000)
OCR_LANGS = env_list("OCR_LANGS", ["eng"])
